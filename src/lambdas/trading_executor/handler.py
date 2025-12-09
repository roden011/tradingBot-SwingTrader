"""
Trading Executor Lambda Handler (Refactored)

Thin orchestration layer that delegates all business logic to service classes.

This replaces the monolithic 3000-line handler with a clean, service-oriented architecture.
The original handler is preserved as handler_legacy.py for reference.

Architecture:
- ExecutionOrchestrator coordinates all services
- PDTService handles PDT logic and EOD exits
- BalanceService manages cash/margin tracking
- MarketDataService handles data fetching with caching
- SystemStateService manages kill switch and circuit breaker
- OrderService handles order execution
- StrategyManager aggregates strategy signals
- PositionEvaluator handles rebalancing decisions
- RiskManager validates all trades

Author: Trading Bot Team
Version: 2.0 (Service-Oriented Architecture)
"""

import sys
import os

# Note: In Lambda, the code is deployed directly to /var/task without the 'src' directory
# So imports work without path manipulation since modules are at the root
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.execution_orchestrator import ExecutionOrchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)


def lambda_handler(event, context):
    """
    AWS Lambda handler for trading execution.

    This is a thin wrapper that delegates to ExecutionOrchestrator.
    All business logic lives in service classes.

    Args:
        event: Lambda event (dict)
        context: Lambda context object

    Returns:
        dict: Lambda response with execution results
            {
                'statusCode': 200 or 500,
                'body': JSON string with results
            }

    Architecture Benefits:
    - Services are independently testable
    - Clear separation of concerns
    - Easy to modify and extend
    - Maintains all performance optimizations
    - Preserves all functionality from original handler
    """
    logger.info("=" * 80)
    logger.info("Trading Executor Lambda Handler v2.0 (Service-Oriented)")
    logger.info("=" * 80)

    try:
        # Create orchestrator instance
        orchestrator = ExecutionOrchestrator()

        # Execute trading cycle
        # All logic delegated to orchestrator and its services
        response = orchestrator.execute(event, context)

        logger.info("Execution completed successfully")
        return response

    except Exception as e:
        # Top-level error handling
        logger.error(f"CRITICAL: Unhandled exception in lambda_handler: {str(e)}", exc_info=True)

        # Return error response
        import json
        from datetime import datetime

        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': 'Internal server error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }


# For local testing
if __name__ == '__main__':
    """
    Local testing entry point

    Usage:
        python handler.py

    Note: Requires AWS credentials and environment variables to be set
    """
    import json

    class MockContext:
        """Mock Lambda context for local testing"""
        def __init__(self):
            self.request_id = 'local-test-12345'
            self.function_name = 'trading-executor-local'
            self.memory_limit_in_mb = 2048
            self.invoked_function_arn = 'arn:aws:lambda:local'

    # Mock event
    test_event = {}

    # Mock context
    test_context = MockContext()

    # Execute
    print("Running local test...")
    result = lambda_handler(test_event, test_context)

    # Print results
    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(json.dumps(json.loads(result['body']), indent=2))
    print("=" * 80)
