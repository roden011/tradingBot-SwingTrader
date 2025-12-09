"""
6-Layer Risk Management System

1. Trade-Level: 2% max risk per trade, 10% max position size, ATR-based stops (2x ATR)
2. Portfolio-Level: 6% max total risk, 30% max sector concentration
3. Time-Based: -2% daily loss limit, -5% weekly loss limit, -10% max drawdown
4. Pattern-Based: Max 3 consecutive losses, 40% min win rate
5. Market Conditions: VIX < 30, 5% max daily volatility
6. Technical Safeguards: Circuit breaker system, API failure monitoring
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of risk checks"""

    passed: bool
    violations: List[str]
    warnings: List[str]
    risk_score: float  # 0.0 to 1.0
    metadata: Dict

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"RiskCheck({status}, violations={len(self.violations)}, score={self.risk_score:.2f})"


class RiskManager:
    """
    6-layer risk management system
    """

    def __init__(self, config: Optional[Dict] = None, use_margin: bool = False):
        """
        Initialize risk manager

        Args:
            config: Risk management configuration dict with nested layer structures
            use_margin: Whether to allow margin trading (default: False)
        """
        config = config or {}

        # Margin control
        self.use_margin = use_margin

        # Layer 1: Trade-Level
        layer1 = config.get('layer1_trade_level', {})
        self.max_risk_per_trade = layer1.get('max_risk_per_trade', 0.02)
        self.max_position_size = layer1.get('max_position_size', 0.10)
        self.atr_stop_multiplier = layer1.get('atr_stop_multiplier', 2.0)

        # Layer 2: Portfolio-Level
        layer2 = config.get('layer2_portfolio_level', {})
        self.max_total_risk = layer2.get('max_total_risk', 0.06)
        self.max_sector_concentration = layer2.get('max_sector_concentration', 0.30)

        # Layer 3: Time-Based
        layer3 = config.get('layer3_time_based', {})
        self.max_daily_loss = layer3.get('max_daily_loss', 0.02)
        self.max_weekly_loss = layer3.get('max_weekly_loss', 0.05)
        self.max_drawdown = layer3.get('max_drawdown', 0.10)

        # Layer 4: Pattern-Based
        layer4 = config.get('layer4_pattern_based', {})
        self.max_consecutive_losses = layer4.get('max_consecutive_losses', 3)
        self.min_win_rate = layer4.get('min_win_rate', 0.40)

        # Layer 5: Market Conditions
        layer5 = config.get('layer5_market_conditions', {})
        self.max_vix = layer5.get('max_vix', 30.0)
        self.max_daily_volatility = layer5.get('max_daily_volatility', 0.05)

        # Layer 6: Technical Safeguards
        layer6 = config.get('layer6_safeguards', {})
        self.circuit_breaker_enabled = layer6.get('circuit_breaker_enabled', False)
        self.kill_switch_enabled = layer6.get('kill_switch_enabled', False)

        logger.info(
            f"Risk manager initialized with 6-layer system: "
            f"max_risk_per_trade={self.max_risk_per_trade:.2%}, "
            f"max_position_size={self.max_position_size:.2%}, "
            f"max_daily_loss={self.max_daily_loss:.2%}, "
            f"use_margin={self.use_margin}"
        )

    def check_trade_risk(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict,
        market_data: pd.DataFrame,
        **kwargs
    ) -> RiskCheckResult:
        """
        Comprehensive risk check for a trade

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Trade price
            portfolio_value: Current portfolio value
            current_positions: Dict of current positions
            market_data: Historical market data
            **kwargs: Additional data (trades_history, risk_metrics, buying_power, etc.)

        Returns:
            RiskCheckResult
        """
        violations = []
        warnings = []
        metadata = {}

        # Layer 0: Cash Availability Check (for buy orders)
        if side == 'buy':
            cash_violations = self._check_cash_availability(
                symbol, quantity, price, kwargs.get('buying_power'), kwargs.get('cash_balance')
            )
            violations.extend(cash_violations)

        # Layer 1: Trade-Level Checks
        trade_violations = self._check_trade_level(
            symbol, side, quantity, price, portfolio_value, market_data, current_positions
        )
        violations.extend(trade_violations)

        # Layer 2: Portfolio-Level Checks
        portfolio_violations = self._check_portfolio_level(
            symbol, side, quantity, price, portfolio_value, current_positions
        )
        violations.extend(portfolio_violations)

        # Layer 3: Time-Based Checks
        time_violations = self._check_time_based(
            portfolio_value, kwargs.get('initial_portfolio_value', portfolio_value),
            kwargs.get('risk_metrics', {})
        )
        violations.extend(time_violations)

        # Layer 4: Pattern-Based Checks
        pattern_violations = self._check_pattern_based(
            kwargs.get('trades_history', []),
            kwargs.get('risk_metrics', {})
        )
        violations.extend(pattern_violations)

        # Layer 5: Market Conditions Checks
        market_violations = self._check_market_conditions(
            market_data, kwargs.get('vix_level', 15.0)
        )
        violations.extend(market_violations)

        # Layer 6: Technical Safeguards
        safeguard_violations = self._check_technical_safeguards(
            kwargs.get('system_state', {})
        )
        violations.extend(safeguard_violations)

        # Calculate risk score (0 = low risk, 1 = high risk)
        risk_score = self._calculate_risk_score(
            violations, warnings, quantity, price, portfolio_value
        )

        # Determine if trade passes
        passed = len(violations) == 0

        metadata = {
            'layers_checked': 6,
            'total_violations': len(violations),
            'total_warnings': len(warnings),
            'risk_score': risk_score,
        }

        return RiskCheckResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
            metadata=metadata
        )

    def _check_cash_availability(
        self, symbol: str, quantity: float, price: float,
        buying_power: Optional[float], cash_balance: Optional[float]
    ) -> List[str]:
        """Layer 0: Cash Availability Check (for buy orders only)"""
        violations = []

        trade_cost = quantity * price

        # Determine which balance to check based on margin settings
        if self.use_margin:
            # Margin enabled: check buying power (includes margin)
            if buying_power is not None:
                if trade_cost > buying_power:
                    violations.append(
                        f"Insufficient buying power: ${trade_cost:,.2f} required but only ${buying_power:,.2f} available"
                    )
            elif cash_balance is not None:
                # Fallback to cash if buying_power not provided
                if trade_cost > cash_balance:
                    violations.append(
                        f"Insufficient cash: ${trade_cost:,.2f} required but only ${cash_balance:,.2f} available"
                    )
            else:
                logger.warning("No buying power or cash balance provided for cash availability check")
        else:
            # Margin disabled: only check cash balance (no borrowing)
            if cash_balance is not None:
                if trade_cost > cash_balance:
                    violations.append(
                        f"Insufficient cash: ${trade_cost:,.2f} required but only ${cash_balance:,.2f} available (margin disabled)"
                    )
            elif buying_power is not None:
                # If only buying_power provided, use it but log a warning
                logger.warning("Only buying_power provided but margin disabled; using it as fallback")
                if trade_cost > buying_power:
                    violations.append(
                        f"Insufficient funds: ${trade_cost:,.2f} required but only ${buying_power:,.2f} available"
                    )
            else:
                logger.warning("No cash balance provided for cash availability check (margin disabled)")

        return violations

    def _check_trade_level(
        self, symbol: str, side: str, quantity: float, price: float,
        portfolio_value: float, market_data: pd.DataFrame,
        current_positions: Dict = None
    ) -> List[str]:
        """Layer 1: Trade-Level Risk Checks"""
        violations = []

        # Position size check (including existing holdings)
        trade_value = quantity * price

        # Get existing position value for this symbol
        existing_value = 0.0
        if current_positions and symbol in current_positions:
            existing_value = current_positions[symbol].get('market_value', 0.0)

        # Total position value = existing + new
        total_position_value = existing_value + trade_value
        position_size_pct = total_position_value / portfolio_value

        if position_size_pct > self.max_position_size:
            if existing_value > 0:
                violations.append(
                    f"Position size {position_size_pct:.1%} (existing: ${existing_value:,.0f} + new: ${trade_value:,.0f}) exceeds max {self.max_position_size:.1%}"
                )
            else:
                violations.append(
                    f"Position size {position_size_pct:.1%} exceeds max {self.max_position_size:.1%}"
                )

        # Risk per trade (using ATR-based stop)
        if not market_data.empty and side == 'buy':
            try:
                from tradingbot_core.utils import TechnicalIndicators
                atr = TechnicalIndicators.atr(
                    market_data['high'],
                    market_data['low'],
                    market_data['close']
                )
                current_atr = atr.iloc[-1]
                stop_distance = current_atr * self.atr_stop_multiplier
                risk_amount = quantity * stop_distance
                risk_pct = risk_amount / portfolio_value

                if risk_pct > self.max_risk_per_trade:
                    violations.append(
                        f"Risk per trade {risk_pct:.1%} exceeds max {self.max_risk_per_trade:.1%}"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate ATR-based risk: {e}")

        return violations

    def _check_portfolio_level(
        self, symbol: str, side: str, quantity: float, price: float,
        portfolio_value: float, current_positions: Dict
    ) -> List[str]:
        """Layer 2: Portfolio-Level Risk Checks"""
        violations = []

        # Calculate total exposure after trade
        total_exposure = sum(pos.get('market_value', 0) for pos in current_positions.values())

        if side == 'buy':
            total_exposure += quantity * price

        total_risk_pct = (total_exposure / portfolio_value) * self.max_risk_per_trade

        if total_risk_pct > self.max_total_risk:
            violations.append(
                f"Total portfolio risk {total_risk_pct:.1%} exceeds max {self.max_total_risk:.1%}"
            )

        # Sector concentration (simplified - assumes we have sector info)
        # For now, we'll skip this as we don't have sector data in kwargs
        # In production, you would check sector concentration here

        return violations

    def _check_time_based(
        self, current_value: float, initial_value: float, risk_metrics: Dict
    ) -> List[str]:
        """Layer 3: Time-Based Risk Checks"""
        violations = []

        # Daily loss limit
        daily_pl_pct = risk_metrics.get('daily_pl_pct', 0.0) / 100
        if daily_pl_pct < -self.max_daily_loss:
            violations.append(
                f"Daily loss {daily_pl_pct:.1%} exceeds limit {-self.max_daily_loss:.1%}"
            )

        # Weekly loss limit
        weekly_pl_pct = risk_metrics.get('weekly_pl_pct', 0.0) / 100
        if weekly_pl_pct < -self.max_weekly_loss:
            violations.append(
                f"Weekly loss {weekly_pl_pct:.1%} exceeds limit {-self.max_weekly_loss:.1%}"
            )

        # Drawdown limit
        max_drawdown_pct = risk_metrics.get('max_drawdown_pct', 0.0) / 100
        if abs(max_drawdown_pct) > self.max_drawdown:
            violations.append(
                f"Drawdown {max_drawdown_pct:.1%} exceeds limit {-self.max_drawdown:.1%}"
            )

        return violations

    def _check_pattern_based(
        self, trades_history: List[Dict], risk_metrics: Dict
    ) -> List[str]:
        """Layer 4: Pattern-Based Risk Checks"""
        violations = []

        # Consecutive losses
        consecutive_losses = risk_metrics.get('consecutive_losses', 0)
        if consecutive_losses >= self.max_consecutive_losses:
            violations.append(
                f"Consecutive losses {consecutive_losses} exceeds max {self.max_consecutive_losses}"
            )

        # Win rate
        win_rate = risk_metrics.get('win_rate', 1.0)
        if win_rate < self.min_win_rate and len(trades_history) >= 10:
            violations.append(
                f"Win rate {win_rate:.1%} below minimum {self.min_win_rate:.1%}"
            )

        return violations

    def _check_market_conditions(
        self, market_data: pd.DataFrame, vix_level: float
    ) -> List[str]:
        """Layer 5: Market Conditions Risk Checks"""
        violations = []

        # VIX check
        if vix_level > self.max_vix:
            violations.append(
                f"VIX {vix_level:.1f} exceeds max {self.max_vix:.1f}"
            )

        # Daily volatility check
        if not market_data.empty:
            try:
                from tradingbot_core.utils import TechnicalIndicators
                hv = TechnicalIndicators.historical_volatility(market_data['close'], 20)
                current_vol = hv.iloc[-1]

                # Convert to daily volatility
                daily_vol = current_vol / (252 ** 0.5)

                if daily_vol > self.max_daily_volatility:
                    violations.append(
                        f"Daily volatility {daily_vol:.1%} exceeds max {self.max_daily_volatility:.1%}"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate volatility: {e}")

        return violations

    def _check_technical_safeguards(self, system_state: Dict) -> List[str]:
        """Layer 6: Technical Safeguards Risk Checks"""
        violations = []

        # Circuit breaker
        if system_state.get('circuit_breaker', False):
            violations.append("Circuit breaker is active")

        # Kill switch
        if system_state.get('kill_switch', False):
            violations.append("Kill switch is active")

        # Trading enabled check
        if not system_state.get('trading_enabled', True):
            violations.append("Trading is disabled")

        return violations

    def _calculate_risk_score(
        self, violations: List[str], warnings: List[str],
        quantity: float, price: float, portfolio_value: float
    ) -> float:
        """Calculate overall risk score"""
        # Base score from violations and warnings
        violation_score = len(violations) * 0.3
        warning_score = len(warnings) * 0.1

        # Position size contribution
        position_size_pct = (quantity * price) / portfolio_value
        size_score = position_size_pct / self.max_position_size * 0.3

        # Combine scores
        total_score = violation_score + warning_score + size_score

        # Clamp to 0-1
        return min(1.0, max(0.0, total_score))

    def calculate_position_size(
        self, symbol: str, price: float, portfolio_value: float,
        atr: float, risk_pct: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk parameters

        Args:
            symbol: Stock symbol
            price: Current price
            portfolio_value: Portfolio value
            atr: Average True Range
            risk_pct: Risk percentage (default: max_risk_per_trade)

        Returns:
            Position size (number of shares)
        """
        if risk_pct is None:
            risk_pct = self.max_risk_per_trade

        # Calculate risk amount
        risk_amount = portfolio_value * risk_pct

        # Calculate stop distance
        stop_distance = atr * self.atr_stop_multiplier

        # Calculate position size
        position_size = risk_amount / stop_distance

        # Limit to max position size
        max_shares = (portfolio_value * self.max_position_size) / price
        position_size = min(position_size, max_shares)

        return int(position_size)

    def should_activate_circuit_breaker(self, risk_metrics: Dict) -> bool:
        """Determine if circuit breaker should be activated"""
        # Activate if any time-based limit is breached
        daily_pl_pct = risk_metrics.get('daily_pl_pct', 0.0) / 100
        weekly_pl_pct = risk_metrics.get('weekly_pl_pct', 0.0) / 100
        max_drawdown_pct = abs(risk_metrics.get('max_drawdown_pct', 0.0)) / 100

        if daily_pl_pct < -self.max_daily_loss:
            return True

        if weekly_pl_pct < -self.max_weekly_loss:
            return True

        if max_drawdown_pct > self.max_drawdown:
            return True

        # Activate if consecutive losses exceeded
        consecutive_losses = risk_metrics.get('consecutive_losses', 0)
        if consecutive_losses >= self.max_consecutive_losses:
            return True

        return False
