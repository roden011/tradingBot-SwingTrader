"""
Base Strategy Class
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    """Trading signal from a strategy"""

    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    reason: str
    metadata: dict

    def __repr__(self):
        return f"Signal({self.action}, confidence={self.confidence:.2f}, reason={self.reason})"


class BaseStrategy(ABC):
    """Base class for all trading strategies"""

    def __init__(self, name: str, weight: float):
        """
        Initialize strategy

        Args:
            name: Strategy name
            weight: Strategy weight in consensus (0.0 to 1.0)
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """
        Generate trading signal for a symbol

        Args:
            symbol: Stock symbol
            data: Historical price data (OHLCV)
            **kwargs: Additional data (positions, market data, etc.)

        Returns:
            Trading signal
        """
        pass

    def validate_data(self, data: pd.DataFrame, min_periods: int = 200) -> bool:
        """
        Validate that data has sufficient history

        Args:
            data: Price data
            min_periods: Minimum required periods

        Returns:
            True if data is valid
        """
        if data is None or data.empty:
            return False

        if len(data) < min_periods:
            return False

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False

        return True
