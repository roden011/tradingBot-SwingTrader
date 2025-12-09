"""
Opening Range Breakout Strategy
Weight: 0.30 (configurable)

Day trading strategy that trades breakouts from the opening range.
Only active during first 2 hours of trading (9:30 AM - 11:30 AM ET).

Indicators: Opening range (9:30-9:45 AM high/low), volume
BUY: Price breaks above opening range high with 2x+ volume
SELL: Price breaks below opening range low
"""
import pandas as pd
from datetime import datetime, time
from strategies.base_strategy import BaseStrategy, Signal
from tradingbot_core.utils import TechnicalIndicators


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Opening Range Breakout day trading strategy"""

    def __init__(self, weight: float = 0.30, config: dict = None):
        """
        Initialize Opening Range Breakout Strategy

        Args:
            weight: Strategy weight (from config)
            config: Strategy-specific configuration parameters
        """
        super().__init__(name="Opening Range Breakout", weight=weight)

        # Use config if provided, otherwise use defaults
        config = config or {}
        self.opening_range_minutes = config.get('opening_range_minutes', 15)
        self.min_breakout_percentage = config.get('min_breakout_percentage', 0.5)
        self.volume_multiplier = config.get('volume_multiplier', 2.0)
        self.trading_window_hours = config.get('trading_window_hours', 2.0)

        # Market hours (ET)
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.opening_range_end = time(9, 45)  # 9:45 AM ET (15 min after open)
        self.trading_window_end = time(11, 30)  # 11:30 AM ET

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """
        Generate opening range breakout signal

        Args:
            symbol: Stock symbol
            data: Daily OHLCV data
            **kwargs: Must contain 'intraday_data' with 5-minute bars for current day

        Returns:
            Trading signal
        """
        # Get intraday data (5-minute bars)
        intraday_data = kwargs.get('intraday_data')

        if intraday_data is None or intraday_data.empty:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"{symbol}: No intraday data available for Opening Range Breakout")
            return Signal(
                action="hold",
                confidence=0.0,
                reason="No intraday data available",
                metadata={}
            )

        # Check if we're in the trading window (9:30 AM - 11:30 AM ET)
        current_time = self._get_current_time_et()

        if not self._is_in_trading_window(current_time):
            return Signal(
                action="hold",
                confidence=0.0,
                reason=f"Outside trading window (9:30-11:30 AM ET), current: {current_time.strftime('%H:%M')}",
                metadata={'current_time': current_time.strftime('%H:%M:%S')}
            )

        # Calculate opening range (9:30-9:45 AM)
        opening_range = self._calculate_opening_range(intraday_data)

        if opening_range is None:
            return Signal(
                action="hold",
                confidence=0.0,
                reason="Opening range not yet established (need data from 9:30-9:45 AM)",
                metadata={}
            )

        or_high, or_low, or_range = opening_range

        # Check if range is wide enough to trade
        range_pct = (or_range / or_low) * 100
        if range_pct < self.min_breakout_percentage:
            return Signal(
                action="hold",
                confidence=0.0,
                reason=f"Opening range too narrow ({range_pct:.2f}% < {self.min_breakout_percentage}%)",
                metadata={
                    'or_high': or_high,
                    'or_low': or_low,
                    'or_range_pct': range_pct
                }
            )

        # Get current price and volume
        current_price = intraday_data['close'].iloc[-1]
        current_volume = intraday_data['volume'].iloc[-1]

        # Calculate average volume from daily data
        if not self.validate_data(data, min_periods=20):
            avg_volume = current_volume  # Fallback if insufficient history
        else:
            avg_volume = TechnicalIndicators.volume_sma(data['volume'], 20).iloc[-1]

        # Volume surge check
        volume_surge = current_volume > (avg_volume * self.volume_multiplier)

        metadata = {
            'price': current_price,
            'or_high': or_high,
            'or_low': or_low,
            'or_range_pct': range_pct,
            'volume': current_volume,
            'avg_volume': avg_volume,
            'volume_surge': volume_surge,
            'current_time': current_time.strftime('%H:%M:%S')
        }

        # BUY Signal: Price breaks above opening range high with volume surge
        if current_price > or_high and volume_surge:
            breakout_pct = ((current_price - or_high) / or_high) * 100
            confidence = self._calculate_buy_confidence(
                breakout_pct, current_volume, avg_volume, range_pct
            )
            return Signal(
                action="buy",
                confidence=confidence,
                reason=f"Breakout above OR high {or_high:.2f} (+{breakout_pct:.1f}%) with volume surge",
                metadata=metadata
            )

        # SELL Signal: Price breaks below opening range low
        if current_price < or_low:
            breakdown_pct = ((or_low - current_price) / or_low) * 100
            confidence = self._calculate_sell_confidence(breakdown_pct, range_pct)
            return Signal(
                action="sell",
                confidence=confidence,
                reason=f"Breakdown below OR low {or_low:.2f} (-{breakdown_pct:.1f}%)",
                metadata=metadata
            )

        # HOLD: Price within opening range
        return Signal(
            action="hold",
            confidence=0.0,
            reason=f"Price within opening range ({or_low:.2f} - {or_high:.2f})",
            metadata=metadata
        )

    def _calculate_opening_range(self, intraday_data: pd.DataFrame) -> tuple:
        """
        Calculate opening range (high/low from 9:30-9:45 AM)

        Args:
            intraday_data: 5-minute bars for current day

        Returns:
            Tuple of (or_high, or_low, or_range) or None if not available
        """
        if intraday_data.empty:
            return None

        # Ensure index is datetime
        if not isinstance(intraday_data.index, pd.DatetimeIndex):
            return None

        # Filter for opening range period (9:30-9:45 AM ET)
        # Note: Assumes intraday_data is already in ET timezone
        opening_bars = intraday_data.between_time('09:30', '09:45')

        if opening_bars.empty or len(opening_bars) < 2:
            # Need at least 2 bars (15 minutes of 5-min bars = 3 bars, but allow 2 minimum)
            return None

        or_high = opening_bars['high'].max()
        or_low = opening_bars['low'].min()
        or_range = or_high - or_low

        return (or_high, or_low, or_range)

    def _get_current_time_et(self) -> time:
        """
        Get current time in Eastern Time

        Note: This is a simplified version. In production, should use pytz for proper timezone handling.
        For now, assumes system time is UTC and converts to ET (UTC-5 or UTC-4 depending on DST).
        """
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)

        # Simple conversion: ET is UTC-5 (EST) or UTC-4 (EDT)
        # This is approximate - in production should use pytz
        # For now, use UTC-5 as default (EST)
        hour_offset = 5

        # Calculate ET hour properly (handle crossing midnight)
        et_hour = (now_utc.hour - hour_offset) % 24
        now_et = now_utc.replace(hour=et_hour)

        # Log the time for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Current time - UTC: {now_utc.time()}, ET: {now_et.time()}")

        return now_et.time()

    def _is_in_trading_window(self, current_time: time) -> bool:
        """
        Check if current time is within trading window (9:30 AM - 11:30 AM ET)

        Args:
            current_time: Current time in ET

        Returns:
            True if in trading window
        """
        return self.market_open <= current_time <= self.trading_window_end

    def _calculate_buy_confidence(
        self, breakout_pct: float, volume: float, avg_volume: float, range_pct: float
    ) -> float:
        """
        Calculate confidence for buy signal

        Args:
            breakout_pct: Percentage breakout above opening range high
            volume: Current volume
            avg_volume: Average volume
            range_pct: Opening range size as percentage

        Returns:
            Confidence score 0.0-1.0
        """
        # Breakout strength (larger breakouts = higher confidence)
        breakout_confidence = min(1.0, breakout_pct / 2.0)  # 2% breakout = max

        # Volume surge strength
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        volume_confidence = min(1.0, (volume_ratio - 1.0) / 2.0)  # 3x volume = max

        # Range size (wider ranges = more significant)
        range_confidence = min(1.0, range_pct / 2.0)  # 2% range = max

        # Weighted average (breakout and volume are most important)
        return (breakout_confidence * 0.5 + volume_confidence * 0.4 + range_confidence * 0.1)

    def _calculate_sell_confidence(self, breakdown_pct: float, range_pct: float) -> float:
        """
        Calculate confidence for sell signal

        Args:
            breakdown_pct: Percentage breakdown below opening range low
            range_pct: Opening range size as percentage

        Returns:
            Confidence score 0.0-1.0
        """
        # Breakdown strength
        breakdown_confidence = min(1.0, breakdown_pct / 2.0)  # 2% breakdown = max

        # Range size
        range_confidence = min(1.0, range_pct / 2.0)  # 2% range = max

        # Weighted average
        return (breakdown_confidence * 0.8 + range_confidence * 0.2)
