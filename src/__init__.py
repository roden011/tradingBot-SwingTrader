"""
TradingBot SwingTrader

PDT-compliant swing trading implementation using tradingbot-core.
Holds positions overnight, minimizes day trades.

SwingTrader Specific:
- PDT-aware exit logic (three-tier EOD)
- Dynamic stop adjustment based on position age
- Rebalancing as primary exit strategy
- Multi-day momentum strategy
- Position builder strategy
"""

__version__ = "0.1.0"
