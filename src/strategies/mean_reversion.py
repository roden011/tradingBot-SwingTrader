"""
Mean Reversion Strategy
Weight: 0.25

Indicators: RSI (14-period), Bollinger Bands (20-period, 2σ)
BUY: RSI < 30 AND price in lower 20% of BB range
SELL: RSI > 70 AND price in upper 20% of BB range
"""
import pandas as pd
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion trading strategy"""

    def __init__(self, weight: float = 0.23, config: dict = None):
        """
        Initialize Mean Reversion Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Mean Reversion", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.rsi_period = config.get('rsi_period', 14)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.bb_lower_threshold = config.get('bb_lower_threshold', 0.20)
        self.bb_upper_threshold = config.get('bb_upper_threshold', 0.80)

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate mean reversion signal"""

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
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)
        upper_bb, middle_bb, lower_bb = TechnicalIndicators.bollinger_bands(
            close, self.bb_period, self.bb_std
        )

        # Get latest values
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_upper = upper_bb.iloc[-1]
        current_middle = middle_bb.iloc[-1]
        current_lower = lower_bb.iloc[-1]

        # Calculate position within BB
        bb_position = TechnicalIndicators.bb_position(
            current_price, current_upper, current_middle, current_lower
        )

        metadata = {
            'rsi': current_rsi,
            'bb_position': bb_position,
            'price': current_price,
            'upper_bb': current_upper,
            'middle_bb': current_middle,
            'lower_bb': current_lower,
        }

        # BUY Signal: RSI < 30 AND price in lower 20% of BB range
        if current_rsi < self.rsi_oversold and bb_position < self.bb_lower_threshold:
            confidence = self._calculate_buy_confidence(current_rsi, bb_position)
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Oversold: RSI={current_rsi:.1f}, BB position={bb_position:.2f}",
                metadata=metadata
            )

        # SELL Signal: RSI > 70 AND price in upper 20% of BB range
        if current_rsi > self.rsi_overbought and bb_position > self.bb_upper_threshold:
            confidence = self._calculate_sell_confidence(current_rsi, bb_position)
            return Signal(
                action="sell",
                confidence=confidence,
                reason=f"Overbought: RSI={current_rsi:.1f}, BB position={bb_position:.2f}",
                metadata=metadata
            )

        # HOLD
        return Signal(
            action="hold",
            confidence=0.0,
            reason="No mean reversion signal",
            metadata=metadata
        )

    def _calculate_buy_confidence(self, rsi: float, bb_position: float) -> float:
        """Calculate confidence for buy signal"""
        # More oversold = higher confidence
        rsi_confidence = (self.rsi_oversold - rsi) / self.rsi_oversold
        rsi_confidence = max(0, min(1, rsi_confidence))

        # Lower in BB range = higher confidence
        bb_confidence = (self.bb_lower_threshold - bb_position) / self.bb_lower_threshold
        bb_confidence = max(0, min(1, bb_confidence))

        # Average the two
        return (rsi_confidence + bb_confidence) / 2

    def _calculate_sell_confidence(self, rsi: float, bb_position: float) -> float:
        """Calculate confidence for sell signal"""
        # More overbought = higher confidence
        rsi_confidence = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
        rsi_confidence = max(0, min(1, rsi_confidence))

        # Higher in BB range = higher confidence
        bb_confidence = (bb_position - self.bb_upper_threshold) / (1 - self.bb_upper_threshold)
        bb_confidence = max(0, min(1, bb_confidence))

        # Average the two
        return (rsi_confidence + bb_confidence) / 2
