"""
Volatility Arbitrage Strategy
Weight: 0.10

Indicators: Historical vs Implied Volatility (ATR-based proxy)
BUY: IV < HV by >15%
SELL: IV > HV by >15%

Note: Since we don't have access to real implied volatility,
we use a proxy based on recent volatility trends
"""
import pandas as pd
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class VolatilityArbitrageStrategy(BaseStrategy):
    """Volatility Arbitrage trading strategy"""

    def __init__(self, weight: float = 0.09, config: dict = None):
        """
        Initialize Volatility Arbitrage Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Volatility Arbitrage", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.hv_period = config.get('hv_period', 20)
        self.iv_proxy_period = config.get('iv_proxy_period', 5)
        self.volatility_threshold = config.get('volatility_threshold', 0.15)

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate volatility arbitrage signal"""

        # Validate data
        if not self.validate_data(data):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Insufficient data",
                metadata={}
            )

        # Calculate historical volatility
        close = data['close']
        hv = TechnicalIndicators.historical_volatility(close, self.hv_period)

        # Use short-term volatility as IV proxy
        iv_proxy = TechnicalIndicators.historical_volatility(close, self.iv_proxy_period)

        # Get latest values
        current_price = close.iloc[-1]
        current_hv = hv.iloc[-1]
        current_iv_proxy = iv_proxy.iloc[-1]

        # Handle NaN
        if pd.isna(current_hv) or pd.isna(current_iv_proxy):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Volatility calculation failed",
                metadata={}
            )

        # Calculate volatility spread
        vol_spread = (current_iv_proxy - current_hv) / current_hv

        metadata = {
            'price': current_price,
            'historical_volatility': current_hv,
            'implied_volatility_proxy': current_iv_proxy,
            'volatility_spread': vol_spread,
        }

        # BUY Signal: IV < HV by >15% (volatility is cheap)
        if vol_spread < -self.volatility_threshold:
            confidence = self._calculate_buy_confidence(vol_spread)
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Volatility underpriced (spread={vol_spread:.1%})",
                metadata=metadata
            )

        # SELL Signal: IV > HV by >15% (volatility is expensive)
        if vol_spread > self.volatility_threshold:
            confidence = self._calculate_sell_confidence(vol_spread)
            return Signal(
                action="sell",
                confidence=confidence,
                reason=f"Volatility overpriced (spread={vol_spread:.1%})",
                metadata=metadata
            )

        # HOLD
        return Signal(
            action="hold",
            confidence=0.0,
            reason="Volatility spread within normal range",
            metadata=metadata
        )

    def _calculate_buy_confidence(self, vol_spread: float) -> float:
        """Calculate confidence for buy signal"""
        # More negative spread = higher confidence
        # -15% = 0.5, -30% = 1.0
        confidence = min(1.0, abs(vol_spread - (-self.volatility_threshold)) / 0.15 + 0.5)
        return confidence

    def _calculate_sell_confidence(self, vol_spread: float) -> float:
        """Calculate confidence for sell signal"""
        # More positive spread = higher confidence
        # +15% = 0.5, +30% = 1.0
        confidence = min(1.0, (vol_spread - self.volatility_threshold) / 0.15 + 0.5)
        return confidence
