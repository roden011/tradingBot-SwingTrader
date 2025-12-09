"""
PDT (Pattern Day Trader) Service

Handles PDT tracking, day trade counting, and PDT-aware EOD exit logic.
Implements PDT growth mode strategy for accounts racing to $25K threshold.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from decimal import Decimal

from utils.pdt_tracker import PDTTracker
from utils.logger import setup_logger
from utils.type_conversion import safe_float, safe_divide, validate_positive

logger = setup_logger(__name__)


class PDTStatus:
    """PDT status data class"""
    def __init__(
        self,
        account_value: float,
        day_trades_count: int,
        day_trades_limit: int,
        pdt_threshold: float,
        remaining_day_trades: int,
        is_pdt_exempt: bool,
        is_pdt_restricted: bool
    ):
        self.account_value = account_value
        self.day_trades_count = day_trades_count
        self.day_trades_limit = day_trades_limit
        self.pdt_threshold = pdt_threshold
        self.remaining_day_trades = remaining_day_trades
        self.is_pdt_exempt = is_pdt_exempt
        self.is_pdt_restricted = is_pdt_restricted

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging and serialization"""
        return {
            'account_value': self.account_value,
            'day_trades_count': self.day_trades_count,
            'day_trades_limit': self.day_trades_limit,
            'pdt_threshold': self.pdt_threshold,
            'remaining_day_trades': self.remaining_day_trades,
            'is_pdt_exempt': self.is_pdt_exempt,
            'is_pdt_restricted': self.is_pdt_restricted
        }


class EODExitDecision:
    """EOD exit decision data class"""
    def __init__(
        self,
        should_exit_all: bool,
        should_exit_selective: bool,
        should_hold_all: bool,
        positions_to_exit: List[str],
        positions_to_hold: List[str],
        exit_reason: str,
        pdt_mode: str
    ):
        self.should_exit_all = should_exit_all
        self.should_exit_selective = should_exit_selective
        self.should_hold_all = should_hold_all
        self.positions_to_exit = positions_to_exit
        self.positions_to_hold = positions_to_hold
        self.exit_reason = exit_reason
        self.pdt_mode = pdt_mode


class PDTService:
    """
    Service for handling PDT-related logic including status tracking,
    day trade counting, and PDT-aware EOD exit decisions.

    PDT Growth Mode Strategy:
    - If account >= $25K: Exit all positions (PDT doesn't apply)
    - If account < $25K and NO day trades remaining: Hold ALL positions overnight
    - If account < $25K with day trades available: Only exit significant losses (>threshold)
    """

    def __init__(self, config: Dict, dynamodb_client=None):
        """
        Initialize PDT service

        Args:
            config: Trading configuration dictionary
            dynamodb_client: DynamoDB client for accessing trades table
        """
        self.config = config
        self.dynamodb = dynamodb_client

        # PDT configuration
        pdt_config = config.get('pdt_tracking', {})
        self.pdt_enabled = pdt_config.get('enabled', True)
        self.pdt_threshold = float(pdt_config.get('threshold', 25000.0))
        self.pdt_limit = int(pdt_config.get('limit', 3))

        # Initialize PDT tracker
        self.pdt_tracker = PDTTracker(
            pdt_threshold=self.pdt_threshold,
            pdt_limit=self.pdt_limit
        )

        # PDT growth mode settings
        execution_config = config.get('execution', {})
        self.pdt_growth_mode = execution_config.get('pdt_growth_mode', False)

        # Intraday exit rules
        intraday_config = config.get('intraday_exit_rules', {})
        self.pdt_aware_mode = intraday_config.get('pdt_aware_mode', False)
        self.pdt_loss_exit_threshold = intraday_config.get('pdt_loss_exit_threshold', -2.0)

        logger.info(
            f"PDTService initialized: enabled={self.pdt_enabled}, threshold=${self.pdt_threshold:,.0f}, "
            f"limit={self.pdt_limit}, growth_mode={self.pdt_growth_mode}, aware_mode={self.pdt_aware_mode}"
        )

    def get_pdt_status(
        self,
        portfolio_value: float,
        recent_trades: List[Dict],
        current_positions: Dict
    ) -> PDTStatus:
        """
        Get current PDT status including day trade count and restrictions.

        Args:
            portfolio_value: Current portfolio value
            recent_trades: Recent trades list (last 7 days)
            current_positions: Current positions dictionary

        Returns:
            PDTStatus object with current status
        """
        if not self.pdt_enabled:
            return PDTStatus(
                account_value=portfolio_value,
                day_trades_count=0,
                day_trades_limit=self.pdt_limit,
                pdt_threshold=self.pdt_threshold,
                remaining_day_trades=self.pdt_limit,
                is_pdt_exempt=True,
                is_pdt_restricted=False
            )

        # Use PDT tracker to check compliance (with dummy symbol to get current count)
        pdt_check = self.pdt_tracker.check_pdt_compliance(
            symbol='_DUMMY',
            side='buy',
            account_value=portfolio_value,
            recent_trades=recent_trades,
            current_positions=current_positions
        )

        remaining_day_trades = self.pdt_limit - pdt_check.day_trades_count
        is_pdt_exempt = portfolio_value >= self.pdt_threshold
        is_pdt_restricted = portfolio_value < self.pdt_threshold

        status = PDTStatus(
            account_value=pdt_check.account_value,
            day_trades_count=pdt_check.day_trades_count,
            day_trades_limit=pdt_check.day_trades_limit,
            pdt_threshold=pdt_check.pdt_threshold,
            remaining_day_trades=remaining_day_trades,
            is_pdt_exempt=is_pdt_exempt,
            is_pdt_restricted=is_pdt_restricted
        )

        logger.info(
            f"PDT Status: {status.day_trades_count}/{status.day_trades_limit} day trades, "
            f"Account: ${portfolio_value:,.2f} {'(PDT Exempt)' if is_pdt_exempt else '(PDT Restricted)'}"
        )

        return status

    def determine_eod_exits(
        self,
        pdt_status: PDTStatus,
        current_positions: Dict,
        db_positions: Dict
    ) -> EODExitDecision:
        """
        Determine which positions to exit at end of day based on PDT status.

        PDT-Aware Logic:
        1. If account >= $25K: Exit all positions (PDT exempt)
        2. If no day trades remaining: Hold ALL positions overnight (avoid PDT violation)
        3. If day trades available: Selectively exit only significant losses

        Args:
            pdt_status: Current PDT status
            current_positions: Current positions from Alpaca
            db_positions: Position details from DynamoDB (for entry prices)

        Returns:
            EODExitDecision with exit strategy
        """
        # Default: not in PDT growth mode - exit all positions
        if not (self.pdt_aware_mode or self.pdt_growth_mode):
            position_symbols = list(current_positions.keys())
            return EODExitDecision(
                should_exit_all=True,
                should_exit_selective=False,
                should_hold_all=False,
                positions_to_exit=position_symbols,
                positions_to_hold=[],
                exit_reason="Normal EOD exit (not in PDT-aware mode)",
                pdt_mode='disabled'
            )

        # PDT-AWARE MODE LOGIC

        # Scenario 1: Account >= $25K - PDT doesn't apply, exit all
        if pdt_status.is_pdt_exempt:
            logger.info(
                f"💰 Account value ${pdt_status.account_value:,.2f} >= $25K - PDT EXEMPT! "
                f"Exiting all positions normally..."
            )
            position_symbols = list(current_positions.keys())
            return EODExitDecision(
                should_exit_all=True,
                should_exit_selective=False,
                should_hold_all=False,
                positions_to_exit=position_symbols,
                positions_to_hold=[],
                exit_reason="PDT exempt (account >= $25K)",
                pdt_mode='exempt'
            )

        # Scenario 2: NO day trades remaining - hold ALL positions overnight
        if pdt_status.remaining_day_trades == 0:
            logger.warning(
                f"⚠️  NO DAY TRADES REMAINING (0/{pdt_status.day_trades_limit}) - "
                f"HOLDING ALL {len(current_positions)} POSITIONS OVERNIGHT!"
            )
            logger.info("Strategy: Preserve capital, avoid PDT violations. Let winners run, hold losers with stops.")

            # Log positions being held
            for symbol, position in current_positions.items():
                entry_price = position.get('avg_entry_price', 0)
                current_price = position['current_price']
                quantity = position['quantity']
                unrealized_pl = position.get('unrealized_pl', 0)
                unrealized_pl_pct = (unrealized_pl / (entry_price * quantity)) * 100 if entry_price > 0 else 0

                logger.info(
                    f"Holding {symbol} overnight: {quantity} shares @ ${current_price:.2f}, "
                    f"P&L: ${unrealized_pl:+,.2f} ({unrealized_pl_pct:+.2f}%)"
                )

            position_symbols = list(current_positions.keys())
            return EODExitDecision(
                should_exit_all=False,
                should_exit_selective=False,
                should_hold_all=True,
                positions_to_exit=[],
                positions_to_hold=position_symbols,
                exit_reason="No day trades remaining - holding all positions overnight",
                pdt_mode='hold_all'
            )

        # Scenario 3: Day trades available - selective exits for significant losses only
        logger.info(
            f"📊 PDT-AWARE MODE: {pdt_status.remaining_day_trades}/{pdt_status.day_trades_limit} day trades remaining, "
            f"account ${pdt_status.account_value:,.2f}"
        )
        logger.info(f"Strategy: Exit only losses >{abs(self.pdt_loss_exit_threshold)}%, hold winners/small losses overnight")

        # Calculate P&L for each position
        positions_with_pl = []
        for symbol, position in current_positions.items():
            entry_price = safe_float(position.get('avg_entry_price', 0), default=0.0, field_name=f"{symbol}.avg_entry_price")

            # Get more accurate entry price from DynamoDB if available
            db_position = db_positions.get(symbol)
            if db_position:
                entry_price = safe_float(db_position.get('avg_entry_price', entry_price), default=entry_price, field_name=f"{symbol}.db_entry_price")

            # Validate entry price is positive
            if not validate_positive(entry_price, field_name=f"{symbol}.entry_price", allow_zero=False):
                logger.warning(f"Skipping {symbol}: Invalid entry price {entry_price}")
                continue

            current_price = safe_float(position.get('current_price', 0), default=0.0, field_name=f"{symbol}.current_price")
            quantity = safe_float(position.get('quantity', 0), default=0.0, field_name=f"{symbol}.quantity")

            # Calculate P&L with safe division
            unrealized_pl = (current_price - entry_price) * quantity
            position_cost = entry_price * quantity
            unrealized_pl_pct = safe_divide(unrealized_pl, position_cost, default=0.0, field_name=f"{symbol}.pl_pct") * 100

            positions_with_pl.append({
                'symbol': symbol,
                'position': position,
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': quantity,
                'unrealized_pl': unrealized_pl,
                'unrealized_pl_pct': unrealized_pl_pct
            })

        # Sort by P&L% ascending (worst losses first)
        positions_with_pl.sort(key=lambda p: p['unrealized_pl_pct'])

        # Determine which positions to exit (significant losses only)
        positions_to_exit_data = []
        for pos_data in positions_with_pl:
            if pos_data['unrealized_pl_pct'] < self.pdt_loss_exit_threshold:
                positions_to_exit_data.append(pos_data)

        # Limit exits to available day trades
        if len(positions_to_exit_data) > pdt_status.remaining_day_trades:
            logger.warning(
                f"Want to exit {len(positions_to_exit_data)} losing positions, "
                f"but only {pdt_status.remaining_day_trades} day trades available. "
                f"Exiting worst {pdt_status.remaining_day_trades}."
            )
            positions_to_exit_data = positions_to_exit_data[:pdt_status.remaining_day_trades]

        positions_to_exit = [p['symbol'] for p in positions_to_exit_data]
        positions_to_hold = [s for s in current_positions.keys() if s not in positions_to_exit]

        logger.info(
            f"Selective EOD exits: {len(positions_to_exit)} positions, "
            f"holding {len(positions_to_hold)} overnight"
        )

        return EODExitDecision(
            should_exit_all=False,
            should_exit_selective=True,
            should_hold_all=False,
            positions_to_exit=positions_to_exit,
            positions_to_hold=positions_to_hold,
            exit_reason=f"Selective PDT exits: losses > {abs(self.pdt_loss_exit_threshold)}%",
            pdt_mode='selective'
        )

    def should_bypass_profit_taking(self) -> bool:
        """
        Check if profit-taking should be bypassed for PDT growth mode.

        In PDT growth mode, we want to let winners run indefinitely to maximize gains.

        Returns:
            True if profit-taking should be bypassed
        """
        if self.pdt_growth_mode:
            logger.info(
                "💎 PDT GROWTH MODE: Profit-taking DISABLED - letting winners run to maximize gains until $25K target"
            )
            return True
        return False

    def should_bypass_trailing_stops(self) -> bool:
        """
        Check if trailing stop activation (from profits) should be bypassed for PDT growth mode.

        Returns:
            True if trailing stops from profits should be bypassed
        """
        return self.pdt_growth_mode

    def should_bypass_breakeven_stops(self) -> bool:
        """
        Check if breakeven stop adjustments should be bypassed for PDT growth mode.

        Returns:
            True if breakeven stops should be bypassed
        """
        return self.pdt_growth_mode
