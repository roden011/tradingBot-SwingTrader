"""
Intraday Relative Strength Strategy
Weight: 0.20 (configurable)

Day trading strategy that buys stocks outperforming SPY on an intraday basis.
Uses 5-minute bars to compare recent performance (1-4 hours).

Indicators: Relative return vs SPY (1-hour lookback), trend filter
BUY: Stock outperforming SPY >1.5% in last hour, both trending up
SELL: Stock underperforming SPY >1.0% or SPY trending down
"""
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class RelativeStrengthIntradayStrategy(BaseStrategy):
    """Intraday Relative Strength trading strategy"""

    def __init__(self, weight: float = 0.20, config: dict = None):
        """
        Initialize Intraday Relative Strength Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Relative Strength Intraday", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.lookback_hours = config.get('lookback_hours', 1)
        self.outperformance_threshold = config.get('outperformance_threshold', 1.5)
        self.underperformance_threshold = config.get('underperformance_threshold', 1.0)
        self.spy_trend_filter = config.get('spy_trend_filter', True)

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """
        Generate intraday relative strength signal

        Args:
            symbol: Stock symbol
            data: Daily OHLCV data (not used much, mainly for validation)
            **kwargs: Must contain 'intraday_data' (stock) and 'spy_intraday_data' (benchmark)

        Returns:
            Trading signal
        """
        # Get intraday data (5-minute bars)
        intraday_data = kwargs.get('intraday_data')
        spy_intraday_data = kwargs.get('spy_intraday_data')

        if intraday_data is None or intraday_data.empty:
            return Signal(
                action="hold",
                confidence=0.0,
                reason="No intraday data available for stock",
                metadata={}
            )

        if spy_intraday_data is None or spy_intraday_data.empty:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"{symbol}: No SPY intraday data available for Relative Strength comparison")
            return Signal(
                action="hold",
                confidence=0.0,
                reason="No SPY intraday data available for benchmark comparison",
                metadata={'error': 'SPY data missing'}
            )

        # Validate data alignment - ensure both have similar timestamps
        stock_latest_time = intraday_data.index[-1] if hasattr(intraday_data.index, '__getitem__') else None
        spy_latest_time = spy_intraday_data.index[-1] if hasattr(spy_intraday_data.index, '__getitem__') else None

        if stock_latest_time and spy_latest_time:
            time_diff = abs((stock_latest_time - spy_latest_time).total_seconds())
            if time_diff > 300:  # More than 5 minutes difference
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"{symbol}: Stock and SPY data timestamps are misaligned by {time_diff}s")
                return Signal(
                    action="hold",
                    confidence=0.0,
                    reason=f"Stock and SPY data timestamps misaligned ({time_diff}s difference)",
                    metadata={'stock_time': str(stock_latest_time), 'spy_time': str(spy_latest_time)}
                )

        # Calculate number of bars for lookback (12 bars per hour with 5-min data)
        lookback_bars = int(self.lookback_hours * 12)

        if len(intraday_data) < lookback_bars or len(spy_intraday_data) < lookback_bars:
            return Signal(
                action="hold",
                confidence=0.0,
                reason=f"Insufficient intraday data (need {lookback_bars} bars for {self.lookback_hours}h lookback)",
                metadata={}
            )

        # Calculate returns over lookback period
        stock_return = self._calculate_return(intraday_data, lookback_bars)
        spy_return = self._calculate_return(spy_intraday_data, lookback_bars)

        if stock_return is None or spy_return is None:
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Return calculation failed",
                metadata={}
            )

        # Calculate relative strength
        relative_strength = stock_return - spy_return

        # Get current prices
        current_stock_price = intraday_data['close'].iloc[-1]
        current_spy_price = spy_intraday_data['close'].iloc[-1]

        # Check trend direction (using last 30 minutes = 6 bars)
        stock_trending_up = self._is_trending_up(intraday_data, bars=6)
        spy_trending_up = self._is_trending_up(spy_intraday_data, bars=6)

        metadata = {
            'price': current_stock_price,
            'stock_return': stock_return,
            'spy_return': spy_return,
            'relative_strength': relative_strength,
            'stock_trending_up': stock_trending_up,
            'spy_trending_up': spy_trending_up,
            'lookback_hours': self.lookback_hours
        }

        # BUY Signal: Stock outperforming SPY significantly
        # AND both stock and SPY trending up (if trend filter enabled)
        if relative_strength > self.outperformance_threshold:
            # Check trend filter
            if self.spy_trend_filter and not spy_trending_up:
                return Signal(
                    action="hold",
                    confidence=0.0,
                    reason=f"Stock outperforming (+{relative_strength:.1f}%) but SPY trending down",
                    metadata=metadata
                )

            if self.spy_trend_filter and not stock_trending_up:
                return Signal(
                    action="hold",
                    confidence=0.0,
                    reason=f"Stock outperforming (+{relative_strength:.1f}%) but stock trending down",
                    metadata=metadata
                )

            confidence = self._calculate_buy_confidence(
                relative_strength, stock_return, spy_return, stock_trending_up, spy_trending_up
            )
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Outperforming SPY by {relative_strength:.1f}% in last {self.lookback_hours}h",
                metadata=metadata
            )

        # SELL Signal: Stock underperforming SPY significantly OR SPY trending down
        if relative_strength < -self.underperformance_threshold:
            confidence = self._calculate_sell_confidence(relative_strength, spy_trending_up)
            return Signal(
                action="sell",
                confidence=confidence,
                reason=f"Underperforming SPY by {abs(relative_strength):.1f}% in last {self.lookback_hours}h",
                metadata=metadata
            )

        # SELL Signal: SPY trending down (market weakness)
        if self.spy_trend_filter and not spy_trending_up and spy_return < -0.5:
            confidence = self._calculate_sell_confidence(relative_strength, spy_trending_up)
            return Signal(
                action="sell",
                confidence=confidence,
                reason=f"SPY trending down ({spy_return:.1f}%), market weakness",
                metadata=metadata
            )

        # HOLD: Relative strength within neutral range
        return Signal(
            action="hold",
            confidence=0.0,
            reason=f"Relative strength neutral ({relative_strength:.1f}%)",
            metadata=metadata
        )

    def _calculate_return(self, intraday_data: pd.DataFrame, lookback_bars: int) -> float:
        """
        Calculate return over lookback period

        Args:
            intraday_data: 5-minute OHLCV bars
            lookback_bars: Number of bars to look back

        Returns:
            Return as percentage, or None if calculation fails
        """
        if len(intraday_data) < lookback_bars:
            return None

        current_price = intraday_data['close'].iloc[-1]
        past_price = intraday_data['close'].iloc[-lookback_bars]

        if past_price == 0:
            return None

        return ((current_price - past_price) / past_price) * 100

    def _is_trending_up(self, intraday_data: pd.DataFrame, bars: int = 6) -> bool:
        """
        Check if price is trending up using simple linear regression

        Args:
            intraday_data: 5-minute OHLCV bars
            bars: Number of bars to check (default 6 = 30 minutes)

        Returns:
            True if trending up (positive slope)
        """
        if len(intraday_data) < bars:
            return False

        recent_prices = intraday_data['close'].iloc[-bars:].values

        # Simple trend check: compare recent average to earlier average
        half = bars // 2
        earlier_avg = np.mean(recent_prices[:half])
        recent_avg = np.mean(recent_prices[half:])

        return recent_avg > earlier_avg

    def _calculate_buy_confidence(
        self, relative_strength: float, stock_return: float, spy_return: float,
        stock_trending_up: bool, spy_trending_up: bool
    ) -> float:
        """
        Calculate confidence for buy signal

        Args:
            relative_strength: Relative strength vs SPY (percentage)
            stock_return: Stock return over lookback period
            spy_return: SPY return over lookback period
            stock_trending_up: Whether stock is trending up
            spy_trending_up: Whether SPY is trending up

        Returns:
            Confidence score 0.0-1.0
        """
        # Relative strength magnitude (higher outperformance = higher confidence)
        rs_confidence = min(1.0, relative_strength / 5.0)  # 5% outperformance = max

        # Absolute stock return (prefer stocks with positive absolute returns)
        stock_return_confidence = min(1.0, max(0.0, stock_return / 3.0))  # 3% gain = max

        # SPY return (prefer when market is also up)
        spy_return_confidence = min(1.0, max(0.0, spy_return / 2.0))  # 2% SPY gain = max

        # Trend alignment bonus
        trend_bonus = 0.0
        if stock_trending_up and spy_trending_up:
            trend_bonus = 0.2

        # Weighted average
        base_confidence = (
            rs_confidence * 0.5 +
            stock_return_confidence * 0.3 +
            spy_return_confidence * 0.2
        )

        return min(1.0, base_confidence + trend_bonus)

    def _calculate_sell_confidence(self, relative_strength: float, spy_trending_up: bool) -> float:
        """
        Calculate confidence for sell signal

        Args:
            relative_strength: Relative strength vs SPY (percentage, negative for underperformance)
            spy_trending_up: Whether SPY is trending up

        Returns:
            Confidence score 0.0-1.0
        """
        # Underperformance magnitude
        underperformance = abs(relative_strength)
        underperf_confidence = min(1.0, underperformance / 5.0)  # 5% underperformance = max

        # Market weakness bonus
        market_weakness_bonus = 0.0 if spy_trending_up else 0.3

        return min(1.0, underperf_confidence + market_weakness_bonus)
