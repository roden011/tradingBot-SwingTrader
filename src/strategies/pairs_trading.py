"""
Pairs Trading Strategy
Weight: 0.15

Indicators: Z-score vs SPY benchmark, 30-period lookback
BUY: Z-score < -2 (underperforming)
SELL: Z-score > 2 (outperforming)
"""
import pandas as pd
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class PairsTradingStrategy(BaseStrategy):
    """Pairs Trading strategy relative to SPY benchmark"""

    def __init__(self, weight: float = 0.13, config: dict = None):
        """
        Initialize Pairs Trading Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Pairs Trading", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.lookback_period = config.get('lookback_period', 30)
        self.z_score_buy_threshold = config.get('z_score_buy_threshold', -2.0)
        self.z_score_sell_threshold = config.get('z_score_sell_threshold', 2.0)

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate pairs trading signal"""

        # Need benchmark data (SPY)
        benchmark_data = kwargs.get('benchmark_data')
        if benchmark_data is None or benchmark_data.empty:
            return Signal(
                action="hold",
                confidence=0.0,
                reason="No benchmark data available",
                metadata={}
            )

        # Validate data
        if not self.validate_data(data, min_periods=self.lookback_period):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Insufficient data",
                metadata={}
            )

        # Calculate z-score relative to benchmark
        close = data['close']
        benchmark_close = benchmark_data['close']

        z_score = TechnicalIndicators.z_score(close, benchmark_close, self.lookback_period)

        # Get latest values
        current_price = close.iloc[-1]
        current_z_score = z_score.iloc[-1]

        # Handle NaN
        if pd.isna(current_z_score):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Z-score calculation failed",
                metadata={}
            )

        metadata = {
            'price': current_price,
            'z_score': current_z_score,
            'benchmark': 'SPY',
        }

        # BUY Signal: Z-score < -2 (underperforming SPY)
        if current_z_score < self.z_score_buy_threshold:
            confidence = self._calculate_buy_confidence(current_z_score)
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Underperforming benchmark (Z={current_z_score:.2f})",
                metadata=metadata
            )

        # SELL Signal: Z-score > 2 (outperforming SPY)
        if current_z_score > self.z_score_sell_threshold:
            confidence = self._calculate_sell_confidence(current_z_score)
            return Signal(
                action="sell",
                confidence=confidence,
                reason=f"Outperforming benchmark (Z={current_z_score:.2f})",
                metadata=metadata
            )

        # HOLD
        return Signal(
            action="hold",
            confidence=0.0,
            reason="Z-score within normal range",
            metadata=metadata
        )

    def _calculate_buy_confidence(self, z_score: float) -> float:
        """Calculate confidence for buy signal"""
        # More negative z-score = higher confidence
        # -2 = 0.5 confidence, -3 = 0.75, -4 = 1.0
        confidence = min(1.0, abs(z_score - self.z_score_buy_threshold) / 4.0 + 0.5)
        return confidence

    def _calculate_sell_confidence(self, z_score: float) -> float:
        """Calculate confidence for sell signal"""
        # More positive z-score = higher confidence
        # +2 = 0.5 confidence, +3 = 0.75, +4 = 1.0
        confidence = min(1.0, (z_score - self.z_score_sell_threshold) / 4.0 + 0.5)
        return confidence
