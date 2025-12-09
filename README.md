# TradingBot SwingTrader

PDT-compliant swing trading bot that holds positions overnight to maximize capital growth while minimizing day trades. Uses intelligent rebalancing to sell weak positions only when better opportunities arise.

## Overview

SwingTrader is designed for PDT-compliant capital growth:
- **Hold winners overnight** - No arbitrary profit-taking, let winners run
- **Minimize day trades** - Strategic use of limited day trades (< $25k accounts)
- **Rebalancing over exits** - Sell weak positions to fund stronger opportunities
- **PDT-aware stops** - Different stop behavior for T+0 vs T+1+ positions

## Architecture

Service-Oriented Architecture v2.0 with clean separation of concerns:

```
src/
├── lambdas/trading_executor/     # Lambda handler (thin wrapper)
├── services/
│   ├── execution_orchestrator.py # Main coordinator
│   ├── pdt_service.py            # PDT tracking, day trade limits
│   └── order_service.py          # Order execution with tax tracking
├── strategies/
│   ├── opening_range_breakout.py
│   ├── relative_strength_intraday.py
│   ├── pairs_trading.py
│   └── strategy_manager.py
├── market_scanner/               # Universe discovery
├── risk_management/              # Risk checks
└── position_management/          # Position evaluation
```

## Core Library Integration

Uses [tradingbot-core](https://github.com/roden011/tradingBot-Core) for shared functionality:

```python
from tradingbot_core import Trade, OrderStatus, OrderResult
from tradingbot_core import PDTTracker, WashSaleTracker
from tradingbot_core import setup_logger, PositionReconciler, load_config_from_s3
from tradingbot_core import safe_float, safe_int, safe_divide, validate_positive
from tradingbot_core.services import BalanceService, SystemStateService, MarketDataService, TaxService
from tradingbot_core.utils import TechnicalIndicators
```

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
│ Losing (< -3%)  │ Any              │ EXIT (if T+0)    │
│ Winning         │ Any              │ HOLD overnight   │
│ Small loss      │ 0                │ HOLD overnight   │
│ Small loss      │ > 0              │ EXIT (optional)  │
└─────────────────┴──────────────────┴──────────────────┘
```

### Rebalancing Strategy
Positions are only sold to **rebalance into better opportunities**:
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
  - Alpaca trailing stop orders (exit = NOT a day trade)
  - Trailing stop activates after +3% gain
```

## Deployment

### Prerequisites
- AWS Account with CDK bootstrapped
- Python 3.10+
- Alpaca paper trading account

### Setup
```bash
cd /Users/willroden/Documents/dev/tradingBot-SwingTrader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Set AWS profile
export AWS_PROFILE=tradingbot-dev
```

### Deploy
```bash
# Deploy blue stage
cdk deploy TradingBot-dev-swing-trader-blue

# Deploy green stage
cdk deploy TradingBot-dev-swing-trader-green
```

### Blue/Green Stages
- `dev-swing-trader-blue` - Active trading stage
- `dev-swing-trader-green` - Standby/testing stage

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

## Configuration

Configuration is stored in S3 and loaded at runtime:
- `config/dev-swing-trader-blue.json`
- `config/dev-swing-trader-green.json`

Key configuration sections:
- `trading` - Consensus threshold, strategy weights
- `position_management` - Stop settings, rebalancing rules
- `risk_management` - Max positions, exposure limits
- `pdt` - Day trade limits, PDT growth mode settings

## Common Commands

```bash
# View logs
aws logs tail /aws/lambda/trading-bot-dev-swing-trader-blue-executor --follow

# Check positions
aws dynamodb scan --table-name trading-bot-dev-swing-trader-blue-positions

# Check day trades
aws dynamodb scan --table-name trading-bot-dev-swing-trader-blue-day-trades

# Invoke kill switch
aws lambda invoke --function-name trading-bot-dev-swing-trader-blue-kill-switch /dev/null

# Synthesize CDK (no deploy)
cdk synth TradingBot-dev-swing-trader-blue
```

## Execution Flow

```
Lambda Invocation (every 10 min)
    │
    ├── Pre-execution checks (market open, system state)
    │
    ├── PHASE 1: PRE-TRADING CHECKS
    │   ├── Get PDT status (day trades used, remaining)
    │   ├── EOD exit check (PDT-aware selective exit)
    │   └── Stop-loss check (software backup)
    │
    ├── PHASE 2: MARKET DATA
    │   ├── Get trading universe
    │   ├── Fetch historical data (300 days)
    │   └── Fetch intraday data (5-min bars)
    │
    ├── PHASE 3: TRADING
    │   ├── Collect BUY/SELL signals
    │   ├── Execute SELL signals
    │   └── Execute BUY signals (strongest first)
    │
    ├── PHASE 4: REBALANCING
    │   ├── Identify weak positions
    │   ├── Identify strong candidates
    │   ├── Sell weakest to fund strongest
    │   └── Track as day trade if same-day
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
# T+0 = opened today (exit = day trade)
# T+1+ = opened previous day (exit = NOT day trade)

entry_date = datetime.fromisoformat(position['entry_timestamp']).date()
today = datetime.now(timezone.utc).date()
position_age_days = (today - entry_date).days
```

## License

MIT
