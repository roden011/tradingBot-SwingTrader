"""
Pattern Day Trader (PDT) Rule Tracker

Tracks day trades to prevent PDT rule violations for accounts <$25K.

PDT Rule:
- If you make 4+ day trades in 5 business days AND account value <$25K
- Account gets flagged and restricted from day trading for 90 days

Day Trade: Buy and sell the same security on the same day
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DayTradeCheck:
    """Result of PDT check"""
    can_trade: bool
    day_trades_count: int
    day_trades_limit: int
    account_value: float
    pdt_threshold: float
    warning_message: Optional[str] = None
    blocking_message: Optional[str] = None

    def __repr__(self):
        if self.can_trade:
            return f"PDTCheck(OK, {self.day_trades_count}/{self.day_trades_limit} day trades)"
        else:
            return f"PDTCheck(BLOCKED, {self.day_trades_count}/{self.day_trades_limit}, {self.blocking_message})"


class PDTTracker:
    """
    Track day trades to enforce PDT rule compliance
    """

    def __init__(self, pdt_threshold: float = 25000.0, pdt_limit: int = 3):
        """
        Initialize PDT tracker

        Args:
            pdt_threshold: Account value threshold for PDT rule ($25K default)
            pdt_limit: Maximum day trades allowed before triggering PDT (3 = allows 3, blocks 4th)
        """
        self.pdt_threshold = pdt_threshold
        self.pdt_limit = pdt_limit

    def check_pdt_compliance(
        self,
        symbol: str,
        side: str,
        account_value: float,
        recent_trades: List[Dict],
        current_positions: Dict[str, Dict]
    ) -> DayTradeCheck:
        """
        Check if proposed trade would violate PDT rule

        Args:
            symbol: Symbol to trade
            side: 'buy' or 'sell'
            account_value: Current account value
            recent_trades: Recent trades (last 5 business days)
            current_positions: Current positions

        Returns:
            DayTradeCheck with compliance status
        """
        # If account >= $25K, PDT rule doesn't apply
        if account_value >= self.pdt_threshold:
            return DayTradeCheck(
                can_trade=True,
                day_trades_count=0,
                day_trades_limit=999,  # Unlimited
                account_value=account_value,
                pdt_threshold=self.pdt_threshold,
                warning_message=f"Account >${self.pdt_threshold:,.0f}: PDT rule does not apply"
            )

        # Count day trades in last 5 business days
        day_trades = self._count_day_trades_in_last_5_days(recent_trades)

        # Check if this trade would create a day trade
        would_be_day_trade = self._would_create_day_trade(
            symbol, side, recent_trades, current_positions
        )

        if would_be_day_trade:
            # This would be a day trade
            new_day_trade_count = day_trades + 1

            if new_day_trade_count > self.pdt_limit:
                # Would violate PDT rule
                return DayTradeCheck(
                    can_trade=False,
                    day_trades_count=day_trades,
                    day_trades_limit=self.pdt_limit,
                    account_value=account_value,
                    pdt_threshold=self.pdt_threshold,
                    blocking_message=(
                        f"PDT RULE VIOLATION: This {side} of {symbol} would be day trade #{new_day_trade_count}. "
                        f"Limit is {self.pdt_limit} day trades in 5 days for accounts <${self.pdt_threshold:,.0f}. "
                        f"Current account value: ${account_value:,.2f}"
                    )
                )
            elif new_day_trade_count == self.pdt_limit:
                # Last allowed day trade - warn
                return DayTradeCheck(
                    can_trade=True,
                    day_trades_count=day_trades,
                    day_trades_limit=self.pdt_limit,
                    account_value=account_value,
                    pdt_threshold=self.pdt_threshold,
                    warning_message=(
                        f"⚠ WARNING: This would be your LAST allowed day trade ({self.pdt_limit}/{self.pdt_limit}). "
                        f"Next day trade will trigger PDT restriction."
                    )
                )
            else:
                # Safe but approaching limit
                return DayTradeCheck(
                    can_trade=True,
                    day_trades_count=day_trades,
                    day_trades_limit=self.pdt_limit,
                    account_value=account_value,
                    pdt_threshold=self.pdt_threshold,
                    warning_message=(
                        f"This will be day trade {new_day_trade_count}/{self.pdt_limit}. "
                        f"{self.pdt_limit - new_day_trade_count} remaining before PDT limit."
                    )
                )
        else:
            # Not a day trade
            return DayTradeCheck(
                can_trade=True,
                day_trades_count=day_trades,
                day_trades_limit=self.pdt_limit,
                account_value=account_value,
                pdt_threshold=self.pdt_threshold
            )

    def _count_day_trades_in_last_5_days(self, recent_trades: List[Dict]) -> int:
        """Count day trades in last 5 business days"""
        cutoff_date = datetime.utcnow() - timedelta(days=7)  # Use 7 days to be safe

        # Group trades by symbol and date
        trades_by_symbol_date: Dict[str, Dict[str, List[Dict]]] = {}

        for trade in recent_trades:
            timestamp = trade.get('timestamp', '')
            if not timestamp:
                logger.warning(f"Trade record missing timestamp, skipping: {trade.get('id', 'unknown')}")
                continue

            try:
                trade_date = datetime.fromisoformat(timestamp).date()
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid timestamp format in trade record: {timestamp}, error: {e}")
                continue

            if datetime.combine(trade_date, datetime.min.time()) < cutoff_date:
                continue

            symbol = trade.get('symbol')
            side = trade.get('side')

            if not symbol or not side:
                continue

            if symbol not in trades_by_symbol_date:
                trades_by_symbol_date[symbol] = {}

            date_str = trade_date.isoformat()
            if date_str not in trades_by_symbol_date[symbol]:
                trades_by_symbol_date[symbol][date_str] = []

            trades_by_symbol_date[symbol][date_str].append(trade)

        # Count day trades (buy and sell on same day)
        day_trade_count = 0

        for symbol, dates in trades_by_symbol_date.items():
            for date_str, trades in dates.items():
                has_buy = any(t.get('side') == 'buy' for t in trades)
                has_sell = any(t.get('side') == 'sell' for t in trades)

                if has_buy and has_sell:
                    day_trade_count += 1
                    logger.debug(f"Day trade detected: {symbol} on {date_str}")

        return day_trade_count

    def _would_create_day_trade(
        self,
        symbol: str,
        side: str,
        recent_trades: List[Dict],
        current_positions: Dict[str, Dict]
    ) -> bool:
        """
        Check if this trade would create a day trade

        Day trade = opening AND closing a position on the same day
        """
        today = datetime.utcnow().date()

        # Get today's trades for this symbol
        today_trades = [
            t for t in recent_trades
            if t.get('symbol') == symbol and
            datetime.fromisoformat(t.get('timestamp', '')).date() == today
        ]

        if side == 'buy':
            # Buying - would create day trade if we sell today
            # Check if we already sold today (and are buying back)
            sold_today = any(t.get('side') == 'sell' for t in today_trades)
            return sold_today

        else:  # sell
            # Selling - would create day trade if we bought today
            bought_today = any(t.get('side') == 'buy' for t in today_trades)

            # Also check if we hold position from earlier (not a day trade)
            has_existing_position = symbol in current_positions and current_positions[symbol].get('quantity', 0) > 0

            # Only day trade if we bought TODAY
            return bought_today
