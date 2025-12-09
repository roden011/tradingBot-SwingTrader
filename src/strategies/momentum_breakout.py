"""
Momentum Breakout Strategy
Weight: 0.25

Indicators: 20-day high/low, volume (1.5x avg), ROC, ATR
BUY: Price breaks above 20-day high with volume surge
SELL: Price breaks below 20-day low OR ROC < -5%
"""
import pandas as pd
from .base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum Breakout trading strategy"""

    def __init__(self, weight: float = 0.23, config: dict = None):
        """
        Initialize Momentum Breakout Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Momentum Breakout", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.breakout_period = config.get('breakout_period', 20)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.roc_period = config.get('roc_period', 12)
        self.roc_threshold = config.get('roc_threshold', -5.0)
        self.atr_period = config.get('atr_period', 14)

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate momentum breakout signal"""

        # Validate data
        if not self.validate_data(data):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Insufficient data",
                metadata={}
            )

        # Calculate indicators
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        rolling_high = TechnicalIndicators.rolling_high(high, self.breakout_period)
        rolling_low = TechnicalIndicators.rolling_low(low, self.breakout_period)
        avg_volume = TechnicalIndicators.volume_sma(volume, self.breakout_period)
        roc = TechnicalIndicators.roc(close, self.roc_period)
        atr = TechnicalIndicators.atr(high, low, close, self.atr_period)

        # Get latest values
        current_price = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_volume = volume.iloc[-1]
        prev_rolling_high = rolling_high.iloc[-2]  # Previous day's high
        prev_rolling_low = rolling_low.iloc[-2]  # Previous day's low
        current_avg_volume = avg_volume.iloc[-1]
        current_roc = roc.iloc[-1]
        current_atr = atr.iloc[-1]

        # Volume surge check
        volume_surge = current_volume > (current_avg_volume * self.volume_multiplier)

        metadata = {
            'price': current_price,
            'rolling_high': prev_rolling_high,
            'rolling_low': prev_rolling_low,
            'volume': current_volume,
            'avg_volume': current_avg_volume,
            'volume_surge': volume_surge,
            'roc': current_roc,
            'atr': current_atr,
        }

        # BUY Signal: Price breaks above 20-day high with volume surge
        if current_price > prev_rolling_high and volume_surge:
            confidence = self._calculate_buy_confidence(
                current_price, prev_rolling_high, current_volume, current_avg_volume, current_roc
            )
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Breakout above {prev_rolling_high:.2f} with volume surge",
                metadata=metadata
            )

        # SELL Signal: Price breaks below 20-day low OR ROC < -5%
        if current_price < prev_rolling_low or current_roc < self.roc_threshold:
            reason = "Breakdown below 20-day low" if current_price < prev_rolling_low else f"Negative momentum (ROC={current_roc:.1f}%)"
            confidence = self._calculate_sell_confidence(
                current_price, prev_rolling_low, current_roc
            )
            return Signal(
                action="sell",
                confidence=confidence,
                reason=reason,
                metadata=metadata
            )

        # HOLD
        return Signal(
            action="hold",
            confidence=0.0,
            reason="No breakout signal",
            metadata=metadata
        )

    def _calculate_buy_confidence(
        self, price: float, rolling_high: float, volume: float, avg_volume: float, roc: float
    ) -> float:
        """Calculate confidence for buy signal"""
        # How far above rolling high
        breakout_strength = (price - rolling_high) / rolling_high
        breakout_confidence = min(1.0, breakout_strength * 50)  # 2% breakout = max

        # Volume surge strength
        volume_ratio = volume / avg_volume
        volume_confidence = min(1.0, (volume_ratio - 1.0) / 2.0)  # 3x volume = max

        # ROC strength
        roc_confidence = min(1.0, max(0, roc / 20))  # 20% ROC = max

        # Weighted average
        return (breakout_confidence * 0.4 + volume_confidence * 0.3 + roc_confidence * 0.3)

    def _calculate_sell_confidence(
        self, price: float, rolling_low: float, roc: float
    ) -> float:
        """Calculate confidence for sell signal"""
        # Breakdown below rolling low
        if price < rolling_low:
            breakdown_strength = (rolling_low - price) / rolling_low
            breakdown_confidence = min(1.0, breakdown_strength * 50)
            return breakdown_confidence

        # Negative ROC
        if roc < self.roc_threshold:
            roc_severity = abs(roc) / 10  # -10% ROC = max confidence
            return min(1.0, roc_severity)

        return 0.5  # Default
