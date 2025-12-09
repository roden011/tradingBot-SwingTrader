"""
Services module for trading bot refactored architecture.

This module contains service classes that handle specific domains of the trading system:

Shared Services (from tradingbot_core):
- BalanceService: Cash/margin tracking and deleveraging
- MarketDataService: Data fetching, caching, parallel operations
- SystemStateService: Kill switch, circuit breaker, state checks
- TaxService: Tax tracking and obligations

Bot-Specific Services (local):
- PDTService: PDT tracking and EOD exit logic
- OrderService: Order creation and submission
- ExecutionOrchestrator: Main orchestration logic

Note: StrategyManager and PositionEvaluator are existing classes used directly.
"""

# Shared services from tradingbot_core
from tradingbot_core.services import (
    BalanceService,
    BalanceTracker,
    MarketDataService,
    SystemStateService,
    TaxService,
    TaxObligation,
)

# Bot-specific services (local)
from services.pdt_service import PDTService
from services.order_service import OrderService
from services.execution_orchestrator import ExecutionOrchestrator, ExecutionContext

__all__ = [
    'PDTService',
    'BalanceService',
    'BalanceTracker',
    'MarketDataService',
    'OrderService',
    'SystemStateService',
    'TaxService',
    'TaxObligation',
    'ExecutionOrchestrator',
    'ExecutionContext',
]
