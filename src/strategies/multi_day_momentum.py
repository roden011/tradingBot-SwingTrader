"""
Multi-Day Momentum Strategy
Weight: 0.35 (configurable)

PDT-optimized strategy for 1-3 day momentum plays.
Identifies stocks with sustained multi-day momentum rather than intraday breakouts.

Indicators: 3-day momentum, 5-day VWAP, RSI (14-period), relative strength vs SPY
BUY: Stock up >2% from 3-day low, above 5-day VWAP, RSI 40-70, outperforming SPY
SELL: Below 3-day trailing low, RSI >80 (overbought), or underperforming SPY by >3%
"""
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class MultiDayMomentumStrategy(BaseStrategy):
    """Multi-day momentum trading strategy optimized for swing trading"""

    def __init__(self, weight: float = 0.35, config: dict = None):
        """
        Initialize Multi-Day Momentum Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Multi Day Momentum", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.lookback_days = config.get('lookback_days', 3)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.rsi_buy_min = config.get('rsi_buy_min', 40)
        self.rsi_buy_max = config.get('rsi_buy_max', 70)
        self.rsi_sell_threshold = config.get('rsi_sell_threshold', 80)
        self.relative_strength_threshold = config.get('relative_strength_threshold', 1.02)
        self.trailing_stop_days = config.get('trailing_stop_days', 3)
        self.vwap_period = 5  # 5-day VWAP

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate multi-day momentum signal"""

        # Validate data
        if not self.validate_data(data, min_periods=60):
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

        # Multi-day momentum
        three_day_low = TechnicalIndicators.rolling_low(low, self.lookback_days)
        three_day_high = TechnicalIndicators.rolling_high(high, self.lookback_days)

        # VWAP
        vwap = TechnicalIndicators.vwap(high, low, close, volume, period=self.vwap_period)

        # RSI
        rsi = TechnicalIndicators.rsi(close, period=14)

        # Volume
        avg_volume = TechnicalIndicators.volume_sma(volume, period=20)

        # Get latest values
        current_price = close.iloc[-1]
        current_three_day_low = three_day_low.iloc[-1]
        current_three_day_high = three_day_high.iloc[-1]
        current_vwap = vwap.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_volume = volume.iloc[-1]
        current_avg_volume = avg_volume.iloc[-1]

        # Calculate momentum from 3-day low
        momentum_from_low = ((current_price - current_three_day_low) / current_three_day_low) * 100

        # Volume surge
        volume_ratio = current_volume / current_avg_volume if current_avg_volume > 0 else 0

        # Relative strength vs SPY (if available)
        spy_data = kwargs.get('spy_data')
        relative_strength = None
        spy_3day_return = None
        stock_3day_return = None

        if spy_data is not None and not spy_data.empty and len(spy_data) >= self.lookback_days:
            # Calculate 3-day returns
            stock_3day_return = ((close.iloc[-1] / close.iloc[-(self.lookback_days + 1)]) - 1) * 100
            spy_3day_return = ((spy_data['close'].iloc[-1] / spy_data['close'].iloc[-(self.lookback_days + 1)]) - 1) * 100
            relative_strength = stock_3day_return / spy_3day_return if spy_3day_return != 0 else 1.0

        metadata = {
            'price': current_price,
            '3day_low': current_three_day_low,
            '3day_high': current_three_day_high,
            'vwap_5d': current_vwap,
            'rsi': current_rsi,
            'momentum_from_low': momentum_from_low,
            'volume_ratio': volume_ratio,
            'relative_strength': relative_strength,
            'stock_3day_return': stock_3day_return,
            'spy_3day_return': spy_3day_return,
        }

        # BUY Signal: Multi-day momentum with confirmation
        buy_conditions = [
            momentum_from_low > 2.0,  # Up >2% from 3-day low
            current_price > current_vwap,  # Above 5-day VWAP
            self.rsi_buy_min <= current_rsi <= self.rsi_buy_max,  # RSI in buy zone
            volume_ratio >= self.volume_threshold,  # Volume surge
        ]

        # Add relative strength condition if SPY data available
        if relative_strength is not None:
            buy_conditions.append(relative_strength >= self.relative_strength_threshold)

        if all(buy_conditions):
            confidence = self._calculate_buy_confidence(
                momentum_from_low,
                current_rsi,
                volume_ratio,
                relative_strength
            )

            rs_text = f", RS vs SPY={relative_strength:.2f}" if relative_strength else ""
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Multi-day momentum: +{momentum_from_low:.1f}% from 3d low, RSI={current_rsi:.1f}{rs_text}",
                metadata=metadata
            )

        # SELL Signal: Loss of momentum or overbought
        sell_conditions = [
            current_price < current_three_day_low,  # Below 3-day trailing low
            current_rsi > self.rsi_sell_threshold,  # Overbought
        ]

        # Add relative strength underperformance check
        if relative_strength is not None and stock_3day_return is not None and spy_3day_return is not None:
            underperformance = stock_3day_return - spy_3day_return
            if underperformance < -3.0:  # Underperforming SPY by >3%
                sell_conditions.append(True)
                metadata['underperformance'] = underperformance

        if any(sell_conditions):
            reason = "Multi-day momentum breakdown"
            if current_price < current_three_day_low:
                reason = f"Below 3-day trailing low ({current_three_day_low:.2f})"
            elif current_rsi > self.rsi_sell_threshold:
                reason = f"Overbought: RSI={current_rsi:.1f}"
            elif metadata.get('underperformance'):
                reason = f"Underperforming SPY by {metadata['underperformance']:.1f}%"

            confidence = self._calculate_sell_confidence(current_rsi, relative_strength)
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
            reason="No multi-day momentum signal",
            metadata=metadata
        )

    def _calculate_buy_confidence(
        self,
        momentum: float,
        rsi: float,
        volume_ratio: float,
        relative_strength: float = None
    ) -> float:
        """Calculate confidence for buy signal"""

        # Stronger momentum = higher confidence (normalize to 0-1)
        momentum_confidence = min(momentum / 10.0, 1.0)  # Cap at 10%

        # RSI in sweet spot (50-60) = higher confidence
        rsi_distance_from_50 = abs(rsi - 50)
        rsi_confidence = max(0, 1 - (rsi_distance_from_50 / 30))  # Best at 50, worst at 20 or 80

        # Higher volume = higher confidence (normalize)
        volume_confidence = min((volume_ratio - 1.0) / 2.0, 1.0)  # 1x = 0, 3x = 1.0

        # Combine confidences
        if relative_strength is not None and relative_strength >= 1.0:
            # Bonus for relative strength
            rs_confidence = min((relative_strength - 1.0) * 2, 0.3)  # Up to 0.3 bonus
            total = (momentum_confidence * 0.4 + rsi_confidence * 0.3 +
                    volume_confidence * 0.3 + rs_confidence)
        else:
            total = (momentum_confidence * 0.4 + rsi_confidence * 0.3 +
                    volume_confidence * 0.3)

        return max(0.0, min(1.0, total))

    def _calculate_sell_confidence(self, rsi: float, relative_strength: float = None) -> float:
        """Calculate confidence for sell signal"""

        # More overbought = higher confidence
        if rsi > self.rsi_sell_threshold:
            rsi_confidence = (rsi - self.rsi_sell_threshold) / (100 - self.rsi_sell_threshold)
        else:
            rsi_confidence = 0.5  # Default for breakdown

        # Relative weakness adds to sell confidence
        if relative_strength is not None and relative_strength < 1.0:
            rs_confidence = (1.0 - relative_strength) * 0.5  # Penalty for weakness
            return max(0.0, min(1.0, (rsi_confidence + rs_confidence) / 1.5))

        return max(0.0, min(1.0, rsi_confidence))
