"""
SwingTrader-Specific Services

These services are specific to swing trading and handle:
- PDT-aware exit logic
- Dynamic stop adjustment by position age
- Rebalancing-based exits

Shared services (BalanceService, MarketDataService, etc.)
are imported from tradingbot_core.services.
"""

# Swing trader specific services will be migrated here:
# from .pdt_exit_service import PDTExitService (or override of PDTService)

__all__ = []
