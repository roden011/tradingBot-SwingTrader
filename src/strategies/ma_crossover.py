"""
Moving Average Crossover Strategy
Weight: 0.35

Indicators: MA20, MA50, MA200
BUY: MA20 crosses above MA50 AND price > MA200
SELL: MA20 crosses below MA50 OR price < MA200
"""
import pandas as pd
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover trading strategy"""

    def __init__(self, weight: float = 0.32, config: dict = None):
        """
        Initialize MA Crossover Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="MA Crossover", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.ma20_period = config.get('ma20_period', 20)
        self.ma50_period = config.get('ma50_period', 50)
        self.ma200_period = config.get('ma200_period', 200)

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate MA crossover signal"""

        # Validate data
        if not self.validate_data(data, min_periods=self.ma200_period):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Insufficient data",
                metadata={}
            )

        # Calculate moving averages
        close = data['close']
        ma20 = TechnicalIndicators.sma(close, self.ma20_period)
        ma50 = TechnicalIndicators.sma(close, self.ma50_period)
        ma200 = TechnicalIndicators.sma(close, self.ma200_period)

        # Get latest values
        current_price = close.iloc[-1]
        current_ma20 = ma20.iloc[-1]
        current_ma50 = ma50.iloc[-1]
        current_ma200 = ma200.iloc[-1]

        # Previous values
        prev_ma20 = ma20.iloc[-2]
        prev_ma50 = ma50.iloc[-2]

        # Detect crossovers
        crossover_signal = TechnicalIndicators.ma_crossover_signal(ma20, ma50)
        current_crossover = crossover_signal.iloc[-1]

        metadata = {
            'price': current_price,
            'ma20': current_ma20,
            'ma50': current_ma50,
            'ma200': current_ma200,
            'crossover': current_crossover,
        }

        # BUY Signal: MA20 crosses above MA50 AND price > MA200
        if current_crossover == 1 and current_price > current_ma200:
            confidence = self._calculate_buy_confidence(
                current_price, current_ma20, current_ma50, current_ma200
            )
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Bullish MA crossover with price > MA200",
                metadata=metadata
            )

        # SELL Signal: MA20 crosses below MA50 OR price < MA200
        if current_crossover == -1 or current_price < current_ma200:
            reason = "Bearish MA crossover" if current_crossover == -1 else "Price < MA200"
            confidence = self._calculate_sell_confidence(
                current_price, current_ma20, current_ma50, current_ma200
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
            reason="No MA crossover signal",
            metadata=metadata
        )

    def _calculate_buy_confidence(
        self, price: float, ma20: float, ma50: float, ma200: float
    ) -> float:
        """Calculate confidence for buy signal"""
        # How far above MA200
        above_ma200_pct = (price - ma200) / ma200
        ma200_confidence = min(1.0, above_ma200_pct * 10)  # 10% above = max confidence

        # How much MA20 is above MA50
        ma_spread = (ma20 - ma50) / ma50
        spread_confidence = min(1.0, ma_spread * 20)  # 5% spread = max confidence

        # Average
        return (ma200_confidence + spread_confidence) / 2

    def _calculate_sell_confidence(
        self, price: float, ma20: float, ma50: float, ma200: float
    ) -> float:
        """Calculate confidence for sell signal"""
        # If bearish crossover
        if ma20 < ma50:
            ma_spread = (ma50 - ma20) / ma50
            spread_confidence = min(1.0, ma_spread * 20)

            # If also below MA200, higher confidence
            if price < ma200:
                below_ma200_pct = (ma200 - price) / ma200
                ma200_confidence = min(1.0, below_ma200_pct * 10)
                return (spread_confidence + ma200_confidence) / 2
            else:
                return spread_confidence * 0.8  # Lower confidence if above MA200

        # If just below MA200
        if price < ma200:
            below_ma200_pct = (ma200 - price) / ma200
            return min(1.0, below_ma200_pct * 10)

        return 0.5  # Default confidence
