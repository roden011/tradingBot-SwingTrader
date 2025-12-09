# CLAUDE.md - TradingBot SwingTrader

This file provides guidance to Claude Code when working in the tradingBot-SwingTrader project.

## Project Overview

**tradingBot-SwingTrader** is a PDT-compliant swing trading bot that holds positions overnight to maximize capital growth while minimizing day trades. It uses intelligent rebalancing to sell weak positions only when better opportunities arise.

**Repository**: https://github.com/roden011/tradingBot-SwingTrader.git
**Python**: 3.10+
**AWS Region**: us-east-2

## Architecture

### Service-Oriented Architecture v2.0

The monolithic handler has been replaced with a clean service-oriented architecture:

```
src/
├── lambdas/
│   └── trading_executor/
│       ├── handler.py          # Thin Lambda wrapper
│       └── handler_legacy.py   # ARCHIVED - old monolithic handler
├── services/
│   ├── execution_orchestrator.py  # Main coordinator (~2400+ lines)
│   ├── pdt_service.py             # PDT tracking, day trade limits
│   └── order_service.py           # Order execution with tax tracking
├── strategies/                    # Local strategy implementations
│   ├── opening_range_breakout.py
│   ├── relative_strength_intraday.py
│   ├── pairs_trading.py
│   └── strategy_manager.py
├── alpaca_client/                 # Alpaca API wrapper
├── market_scanner/                # Universe discovery
├── risk_management/               # Risk checks
├── position_management/           # Position evaluation
├── models/                        # Local models
└── utils/                         # Utilities
```

### Core Library Integration

Uses `tradingbot-core` for shared functionality:
```python
# Models
from tradingbot_core import Trade
from tradingbot_core.models.utils import convert_floats_to_decimal

# Services
from tradingbot_core.services import BalanceService, SystemStateService, MarketDataService, TaxService

# Compliance (PDT and Wash Sale tracking)
from tradingbot_core import PDTTracker, WashSaleTracker, DayTradeRepository

# Utils
from tradingbot_core.utils import TechnicalIndicators
from tradingbot_core import setup_logger, PositionReconciler
from tradingbot_core import load_config_from_s3, ConfigLoader
from tradingbot_core import safe_float, safe_int, safe_divide, validate_positive
```

Note: Local `utils/` files (pdt_tracker.py, wash_sale_tracker.py, type_conversion.py, etc.) are deprecated.
All shared utilities now import from `tradingbot_core`.

## Key Features

### PDT Growth Mode
The core philosophy is to **minimize day trades** while **maximizing overnight gains**:
- Hold winners overnight for unlimited upside
- Only exit losers at EOD to preserve day trades
- Use remaining day trade allowance strategically

### EOD Exit Strategy (PDT-Aware)
Unlike DayTrader (which closes ALL positions), SwingTrader uses selective exits:

```
End of Day Decision Matrix:
┌─────────────────┬──────────────────┬──────────────────┐
│ Position Status │ Day Trades Left  │ Action           │
├─────────────────┼──────────────────┼──────────────────┤
│ Losing (< -3%)  │ Any              │ EXIT (if opened today) │
│ Winning         │ Any              │ HOLD overnight   │
│ Small loss      │ 0                │ HOLD overnight   │
│ Small loss      │ > 0              │ EXIT (optional)  │
└─────────────────┴──────────────────┴──────────────────┘
```

### Rebalancing Strategy
Instead of profit-taking, positions are only sold to **rebalance into better opportunities**:
- Identify weak positions (underperformers)
- Identify strong buy candidates
- Sell weakest to fund strongest
- Maintains full investment with optimal allocation

### No Profit-Taking
- Winners run indefinitely (no take-profit ladder)
- Exit only via:
  - Stop-loss (protection)
  - Rebalancing (better opportunity)
  - PDT EOD exit (losing positions)

### Dynamic Stop Management (PDT-Aware)
Stops are managed based on **position age** to avoid unnecessary day trades:
```
Entry Day (T+0):
  - Software stops only (no Alpaca orders to avoid day trade on exit)
  - Stop at -3% from entry

Next Day (T+1+):
  - Alpaca trailing stop orders (can exit without day trade penalty)
  - Trailing stop activates after +3% gain
```

## Deployment

### Blue/Green Stages
- `dev-swing-trader-blue` - Active trading stage
- `dev-swing-trader-green` - Standby/testing stage

### Deploy Commands
```bash
cd /Users/willroden/Documents/dev/tradingBot-SwingTrader

# Set AWS profile
export AWS_PROFILE=tradingbot-dev

# Activate venv
source venv/bin/activate

# Deploy blue stage
cdk deploy TradingBotStack-dev-swing-trader-blue

# Deploy green stage
cdk deploy TradingBotStack-dev-swing-trader-green
```

### Configuration
- **Config files**: `config/dev-swing-trader-blue.json`, `config/dev-swing-trader-green.json`
- **Secrets**: `trading-bot/alpaca-credentials-dev-swing-trader-{blue|green}` in Secrets Manager
- **Parameters**: `/trading-bot/dev/swing-trader/{blue|green}/use_margin` in SSM

## AWS Resources

### DynamoDB Tables
- `trading-bot-dev-swing-trader-{stage}-positions` - Current positions
- `trading-bot-dev-swing-trader-{stage}-trades` - Trade history
- `trading-bot-dev-swing-trader-{stage}-risk-metrics` - Risk snapshots
- `trading-bot-dev-swing-trader-{stage}-system-state` - Kill switch, circuit breaker
- `trading-bot-dev-swing-trader-{stage}-day-trades` - PDT tracking
- `trading-bot-dev-swing-trader-{stage}-tax-obligations` - Tax tracking

### Lambda Functions
- `trading-bot-dev-swing-trader-{stage}-executor` - Main trading (10-min interval)
- `trading-bot-dev-swing-trader-{stage}-kill-switch` - Emergency stop
- `trading-bot-dev-swing-trader-{stage}-daily-report` - End of day report
- `trading-bot-dev-swing-trader-{stage}-weekly-report` - Weekly summary
- `trading-bot-dev-swing-trader-{stage}-monthly-report` - Monthly analysis
- `trading-bot-dev-swing-trader-{stage}-quarterly-report` - Quarterly review

## Known Issues to Address

### CRITICAL
1. **Missing Lambda handlers** - `kill_switch` and `test_numpy_lambda` handlers don't exist but CDK references them
2. **Silent exception swallowing** in execution_orchestrator.py:2009 - needs logging

### HIGH
1. **DST not handled** in EventBridge schedule (always uses EST offset)
2. **Lazy imports inside methods** - Should move to module level
3. **Missing bounds checking** on `.iloc[-1]` operations
4. **PDT dummy symbol** `'_DUMMY'` could conflict with real symbols

### MEDIUM
1. **Lambda layer path** - `lambda_layer/` directory may not exist
2. **Legacy handler present** - `handler_legacy.py` should be archived
3. **Assert used for validation** - Should use explicit exceptions
4. **Hardcoded secret name** in `submit_stop_orders.py`

## Development Setup

```bash
cd /Users/willroden/Documents/dev/tradingBot-SwingTrader

# Create/activate venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Set AWS profile
export AWS_PROFILE=tradingbot-dev
```

## Common Commands

```bash
# View logs (blue stage)
aws logs tail /aws/lambda/trading-bot-dev-swing-trader-blue-executor --follow

# Check positions table
aws dynamodb scan --table-name trading-bot-dev-swing-trader-blue-positions

# Invoke kill switch
aws lambda invoke --function-name trading-bot-dev-swing-trader-blue-kill-switch /dev/null

# Synthesize CDK (no deploy)
cdk synth TradingBotStack-dev-swing-trader-blue

# Check day trades in past 5 days
aws dynamodb scan --table-name trading-bot-dev-swing-trader-blue-day-trades
```

## Execution Flow

```
Lambda Invocation (every 10 min)
    │
    ▼
ExecutionOrchestrator.execute()
    │
    ├── Pre-execution checks (market open, system state)
    │
    ├── PHASE 1: PRE-TRADING CHECKS
    │   ├── Get PDT status (day trades used, remaining)
    │   ├── EOD exit check (PDT-aware selective exit)
    │   │   ├── Exit losing positions opened today
    │   │   └── Hold winning positions overnight
    │   └── Stop-loss check (software backup)
    │
    ├── PHASE 2: MARKET DATA PREPARATION
    │   ├── Get trading universe (MarketScanner)
    │   ├── Fetch historical data (300 days)
    │   ├── Fetch intraday data (5-min bars, cached)
    │   └── Apply pre-filter
    │
    ├── PHASE 3: MAIN TRADING LOOP (Two-Pass)
    │   ├── Pass 1: Collect BUY/SELL signals
    │   ├── Pass 2: Execute SELL signals (free up cash)
    │   └── Pass 3: Execute BUY signals (strongest first)
    │
    ├── PHASE 4: REBALANCING
    │   ├── Identify weak positions (candidates to sell)
    │   ├── Identify strong candidates (to buy)
    │   ├── Sell weakest to fund strongest
    │   └── Track rebalance as day trade if same-day
    │
    └── PHASE 5: POST-TRADING
        ├── Update position tracking
        ├── Update risk metrics
        └── Check circuit breaker
```

## PDT Logic Details

### Day Trade Counting
A day trade occurs when buying and selling the same security on the same day:
- SwingTrader tracks day trades in DynamoDB
- Maximum 3 day trades in rolling 5-day period (for < $25k accounts)
- PDT Growth Mode aims for 0-1 day trades per week

### Position Age Tracking
```python
# Each position tracks entry timestamp
entry_timestamp = position.get('entry_timestamp')
entry_date = datetime.fromisoformat(entry_timestamp).date()
today = datetime.now(timezone.utc).date()
position_age_days = (today - entry_date).days

# T+0 = opened today (exit = day trade)
# T+1+ = opened previous day (exit = NOT day trade)
```

## File Navigation

When asked about specific functionality:
- "execution flow" → `services/execution_orchestrator.py`
- "PDT tracking" → `services/pdt_service.py` (uses `tradingbot_core.PDTTracker`)
- "order execution" → `services/order_service.py`
- "rebalancing" → `services/execution_orchestrator.py` (search `_execute_rebalancing`)
- "EOD exits" → `services/execution_orchestrator.py` (search `_check_end_of_day_exit`)
- "strategies" → `strategies/` directory
- "CDK/infrastructure" → `infrastructure/trading_bot_stack.py`
- "config" → `config/dev-swing-trader-*.json`

Note: PDT tracker, wash sale tracker, position reconciliation, and type conversion
utilities now live in `tradingbot_core`. Local copies in `utils/` are deprecated.
