"""
Position Builder Strategy
Weight: 0.25 (configurable)

Accumulates positions in trending stocks over multiple entries.
Buys pullbacks in confirmed uptrends, holds through multi-day moves.

Indicators: 20-day MA, 50-day MA, 10-day low, MACD histogram, ATR
BUY: Uptrend confirmed (price > 20MA > 50MA), pullback to support, MACD turning positive
SELL: Trend break (price < 20MA for 2+ days), stop loss, or rebalancing opportunity
"""
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class PositionBuilderStrategy(BaseStrategy):
    """Position building strategy for swing trading"""

    def __init__(self, weight: float = 0.25, config: dict = None):
        """
        Initialize Position Builder Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Position Builder", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.trend_ma_fast = config.get('trend_ma_fast', 20)
        self.trend_ma_slow = config.get('trend_ma_slow', 50)
        self.pullback_threshold = config.get('pullback_threshold', 0.02)  # 2%
        self.volume_pullback_ratio = config.get('volume_pullback_ratio', 0.8)
        self.stop_atr_multiplier = config.get('stop_atr_multiplier', 2.5)
        self.max_position_entries = config.get('max_position_entries', 3)
        self.trend_break_days = 2  # Days below 20MA to trigger sell

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate position building signal"""

        # Validate data
        if not self.validate_data(data, min_periods=self.trend_ma_slow + 20):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Insufficient data",
                metadata={}
            )

        # Get position info if available
        current_position = kwargs.get('position')
        entry_price = current_position.get('entry_price') if current_position else None
        position_entries = current_position.get('entries', 0) if current_position else 0

        # Calculate indicators
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # Moving averages for trend
        ma_20 = TechnicalIndicators.sma(close, self.trend_ma_fast)
        ma_50 = TechnicalIndicators.sma(close, self.trend_ma_slow)

        # Support/resistance
        ten_day_low = TechnicalIndicators.rolling_low(low, 10)
        ten_day_high = TechnicalIndicators.rolling_high(high, 10)

        # MACD
        macd_line, signal_line, macd_hist = TechnicalIndicators.macd(close)

        # ATR for stops
        atr = TechnicalIndicators.atr(high, low, close, period=14)

        # Volume
        avg_volume = TechnicalIndicators.volume_sma(volume, period=20)

        # Get latest values with NaN safety
        current_price = close.iloc[-1]
        current_ma20 = ma_20.iloc[-1]
        current_ma50 = ma_50.iloc[-1]
        current_10d_low = ten_day_low.iloc[-1]
        current_10d_high = ten_day_high.iloc[-1]
        current_macd_hist = macd_hist.iloc[-1]
        prev_macd_hist = macd_hist.iloc[-2] if len(macd_hist) >= 2 else 0
        current_atr = atr.iloc[-1]
        current_volume = volume.iloc[-1]
        current_avg_volume = avg_volume.iloc[-1]

        # Check for NaN values in critical indicators
        if pd.isna(current_ma20) or pd.isna(current_ma50) or pd.isna(current_atr):
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Indicator calculation returned NaN",
                metadata={}
            )

        # Check trend status
        in_uptrend = (current_price > current_ma20 and current_ma20 > current_ma50)

        # Calculate distance from 10-day low (for pullback detection)
        distance_from_low = ((current_price - current_10d_low) / current_10d_low) if current_10d_low > 0 else 0

        # Volume behavior on pullback
        volume_ratio = current_volume / current_avg_volume if current_avg_volume > 0 else 1.0

        # MACD turning positive
        macd_turning_positive = (current_macd_hist > 0 and prev_macd_hist <= 0) or (current_macd_hist > prev_macd_hist and current_macd_hist > -0.1)

        # Days below MA20 (for sell signal)
        days_below_ma20 = self._count_days_below_ma(close, ma_20)

        metadata = {
            'price': current_price,
            'ma_20': current_ma20,
            'ma_50': current_ma50,
            'in_uptrend': in_uptrend,
            '10d_low': current_10d_low,
            'distance_from_low': distance_from_low * 100,  # As percentage
            'macd_hist': current_macd_hist,
            'macd_turning_positive': macd_turning_positive,
            'volume_ratio': volume_ratio,
            'atr': current_atr,
            'days_below_ma20': days_below_ma20,
            'position_entries': position_entries,
        }

        # BUY Signal: Pullback in uptrend with confirmation
        buy_conditions = [
            in_uptrend,  # Confirmed uptrend
            distance_from_low <= self.pullback_threshold,  # Near 10-day low (pullback)
            volume_ratio <= self.volume_pullback_ratio,  # Decreasing volume on pullback
            macd_turning_positive,  # MACD showing momentum return
            position_entries < self.max_position_entries,  # Not at max position size
        ]

        if all(buy_conditions):
            confidence = self._calculate_buy_confidence(
                distance_from_low,
                volume_ratio,
                current_macd_hist,
                in_uptrend
            )

            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Pullback buy in uptrend: {distance_from_low*100:.1f}% from 10d low, MACD turning",
                metadata=metadata
            )

        # SELL Signal: Trend break or stop loss
        if entry_price is not None:
            # Calculate stop loss level
            stop_loss_level = entry_price - (self.stop_atr_multiplier * current_atr)
            metadata['stop_loss_level'] = stop_loss_level
            metadata['entry_price'] = entry_price

            sell_conditions = []

            # Trend break: Below 20MA for multiple days
            if days_below_ma20 >= self.trend_break_days:
                sell_conditions.append(('trend_break', f"Below 20MA for {days_below_ma20} days"))

            # Stop loss
            if current_price < stop_loss_level:
                loss_pct = ((stop_loss_level - current_price) / entry_price) * 100
                sell_conditions.append(('stop_loss', f"Stop loss hit: -{loss_pct:.1f}%"))

            if sell_conditions:
                sell_type, reason = sell_conditions[0]
                confidence = 0.8 if sell_type == 'stop_loss' else 0.6

                return Signal(
                    action="sell",
                    confidence=confidence,
                    reason=reason,
                    metadata=metadata
                )

        # Check for sell without position (shouldn't happen, but handle it)
        if not in_uptrend and days_below_ma20 >= self.trend_break_days:
            return Signal(
                action="sell",
                confidence=0.5,
                reason=f"Downtrend: {days_below_ma20} days below 20MA",
                metadata=metadata
            )

        # HOLD
        return Signal(
            action="hold",
            confidence=0.0,
            reason="No position building signal" if not current_position else "Holding position in trend",
            metadata=metadata
        )

    def _calculate_buy_confidence(
        self,
        distance_from_low: float,
        volume_ratio: float,
        macd_hist: float,
        in_uptrend: bool
    ) -> float:
        """Calculate confidence for buy signal"""

        # Closer to low = higher confidence (inverse relationship)
        distance_confidence = max(0, 1 - (distance_from_low / self.pullback_threshold))

        # Lower volume = higher confidence (healthy pullback)
        volume_confidence = max(0, 1 - volume_ratio) if volume_ratio < 1.0 else 0.5

        # Stronger MACD momentum = higher confidence
        macd_confidence = min(abs(macd_hist) * 2, 1.0) if macd_hist > 0 else 0.5

        # Combine confidences
        total = (distance_confidence * 0.35 +
                volume_confidence * 0.30 +
                macd_confidence * 0.35)

        return max(0.0, min(1.0, total))

    def _count_days_below_ma(self, close: pd.Series, ma: pd.Series, lookback: int = 5) -> int:
        """Count consecutive days price has been below moving average"""
        below_ma = close < ma
        count = 0

        for i in range(min(lookback, len(below_ma))):
            if below_ma.iloc[-(i+1)]:
                count += 1
            else:
                break

        return count
