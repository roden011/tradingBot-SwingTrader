"""
Trading Executor Lambda Handler
Orchestrates strategy execution, risk checks, and trade execution
"""
import os
import json
import logging
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import pickle
import time

# Add Lambda layer path
sys.path.insert(0, '/opt/python')

# Add src directory to path (Lambda root is src/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from alpaca_client.client import AlpacaClient
from strategies.strategy_manager import StrategyManager
from risk_management.risk_manager import RiskManager
from market_scanner.scanner import MarketScanner
from position_management.position_evaluator import PositionEvaluator
from tradingbot_core import Position
from tradingbot_core import Trade
from models.risk_metrics import RiskMetrics
from models.system_state import SystemState
from tradingbot_core import OrderResult, OrderStatus, RejectionReason
from tradingbot_core.utils import TechnicalIndicators
from utils.position_reconciliation import PositionReconciler
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from utils.pdt_tracker import PDTTracker
from utils.wash_sale_tracker import WashSaleTracker

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
sns = boto3.client('sns')
secrets_manager = boto3.client('secretsmanager')
ssm = boto3.client('ssm')

# Environment variables
POSITIONS_TABLE = os.environ['POSITIONS_TABLE']
TRADES_TABLE = os.environ['TRADES_TABLE']
RISK_METRICS_TABLE = os.environ['RISK_METRICS_TABLE']
SYSTEM_STATE_TABLE = os.environ['SYSTEM_STATE_TABLE']
DAY_TRADES_TABLE = os.environ.get('DAY_TRADES_TABLE', '')  # For PDT tracking
REALIZED_LOSSES_TABLE = os.environ.get('REALIZED_LOSSES_TABLE', '')  # For wash sale tracking
DATA_BUCKET = os.environ['DATA_BUCKET']
ALERT_TOPIC_ARN = os.environ['ALERT_TOPIC_ARN']
ALPACA_SECRET_NAME = os.environ['ALPACA_SECRET_NAME']
USE_MARGIN_PARAMETER_NAME = os.environ['USE_MARGIN_PARAMETER_NAME']
TRADING_CONFIG = os.environ.get('TRADING_CONFIG', '{}')  # Full trading configuration as JSON
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')  # dev or prod
STAGE = os.environ.get('STAGE', 'blue')  # blue or green

# Intraday data cache (persists across Lambda warm starts)
# Cache structure: {cache_key: {'data': Dict, 'timestamp': float, 'latest_bar_time': datetime}}
INTRADAY_CACHE = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cache_key(symbols: List[str], start: datetime) -> str:
    """Generate cache key for intraday data"""
    # Round start time to 5 minutes for consistent caching
    rounded_minute = (start.minute // 5) * 5
    rounded_start = start.replace(minute=rounded_minute, second=0, microsecond=0)
    symbols_hash = hash(tuple(sorted(symbols)))
    return f"{symbols_hash}_{rounded_start.isoformat()}"


def get_cached_intraday_data(
    alpaca_client: AlpacaClient,
    symbols: List[str],
    start: datetime,
    end: datetime,
    cache_enabled: bool = True
) -> Dict:
    """
    Get intraday data with caching support

    On cache hit (within TTL):
    - Returns cached data + fetches only the latest bar and appends

    On cache miss:
    - Fetches all data and caches it

    Args:
        alpaca_client: Alpaca client instance
        symbols: List of symbols
        start: Start datetime
        end: End datetime
        cache_enabled: Whether to use caching (default True)

    Returns:
        Dict mapping symbol to DataFrame with intraday bars
    """
    global INTRADAY_CACHE

    if not cache_enabled:
        logger.info("Intraday cache disabled, fetching all data")
        return alpaca_client.get_historical_bars(
            symbols, start, end, timeframe=TimeFrame(5, TimeFrameUnit.Minute)
        )

    cache_key = _get_cache_key(symbols, start)
    current_time = time.time()

    # Check cache
    if cache_key in INTRADAY_CACHE:
        cached_entry = INTRADAY_CACHE[cache_key]
        cache_age = current_time - cached_entry['timestamp']

        if cache_age < CACHE_TTL_SECONDS:
            logger.info(f"Intraday cache HIT (age: {cache_age:.1f}s, TTL: {CACHE_TTL_SECONDS}s)")

            # Get cached data
            cached_data = cached_entry['data']
            latest_bar_time = cached_entry.get('latest_bar_time')

            if latest_bar_time:
                # Fetch only new bars since last cached bar
                new_start = latest_bar_time + timedelta(minutes=5)

                if new_start < end:
                    logger.info(f"Fetching incremental bars from {new_start} to {end}")
                    new_bars = alpaca_client.get_historical_bars(
                        symbols, new_start, end, timeframe=TimeFrame(5, TimeFrameUnit.Minute)
                    )

                    # Append new bars to cached data
                    import pandas as pd
                    for symbol in symbols:
                        if symbol in new_bars and not new_bars[symbol].empty:
                            if symbol in cached_data and not cached_data[symbol].empty:
                                # Concatenate old and new data
                                cached_data[symbol] = pd.concat([cached_data[symbol], new_bars[symbol]])
                                # Remove duplicates (in case of overlap)
                                cached_data[symbol] = cached_data[symbol][~cached_data[symbol].index.duplicated(keep='last')]
                            else:
                                cached_data[symbol] = new_bars[symbol]

                    # Update latest bar time
                    if new_bars:
                        # Get all non-empty DataFrames' max times
                        max_times = [df.index.max() for df in new_bars.values() if not df.empty]
                        if max_times:
                            max_time = max(max_times)
                            cached_entry['latest_bar_time'] = max_time
                else:
                    logger.info("No new bars needed, using cached data")

            cached_entry['timestamp'] = current_time  # Refresh cache timestamp
            return cached_data
        else:
            logger.info(f"Intraday cache EXPIRED (age: {cache_age:.1f}s > TTL: {CACHE_TTL_SECONDS}s)")

    # Cache miss or expired - fetch all data
    logger.info("Intraday cache MISS, fetching all data")
    intraday_data = alpaca_client.get_historical_bars(
        symbols, start, end, timeframe=TimeFrame(5, TimeFrameUnit.Minute)
    )

    # Cache the result
    if intraday_data:
        # Find the latest bar time across all symbols
        max_time = None
        for df in intraday_data.values():
            if not df.empty and (max_time is None or df.index.max() > max_time):
                max_time = df.index.max()

        INTRADAY_CACHE[cache_key] = {
            'data': intraday_data,
            'timestamp': current_time,
            'latest_bar_time': max_time
        }
        logger.info(f"Cached intraday data for {len(symbols)} symbols (latest bar: {max_time})")

    # Clean old cache entries (keep only last 3)
    if len(INTRADAY_CACHE) > 3:
        oldest_key = min(INTRADAY_CACHE.keys(), key=lambda k: INTRADAY_CACHE[k]['timestamp'])
        del INTRADAY_CACHE[oldest_key]
        logger.debug("Cleaned oldest cache entry")

    return intraday_data


def load_config() -> Dict:
    """Load trading configuration from environment variable"""
    try:
        config = json.loads(TRADING_CONFIG)
        logger.info(f"Loaded trading config: {len(config)} sections")
        return config
    except Exception as e:
        logger.error(f"Failed to load trading config: {e}")
        return {}


def get_use_margin_setting() -> bool:
    """
    Get use_margin setting from AWS Systems Manager Parameter Store

    Returns:
        bool: True if margin trading is enabled, False otherwise
    """
    try:
        response = ssm.get_parameter(Name=USE_MARGIN_PARAMETER_NAME)
        value = response['Parameter']['Value'].lower().strip()

        # Convert string to boolean
        use_margin = value == 'true'

        logger.info(f"Margin trading setting: use_margin={use_margin} (from Parameter Store)")
        return use_margin

    except Exception as e:
        logger.error(f"Error getting use_margin parameter from Parameter Store: {e}")
        logger.warning("Defaulting to use_margin=False (margin disabled)")
        return False


def get_and_increment_execution_counter() -> int:
    """
    Get the current execution counter from DynamoDB and increment it

    Returns:
        int: Current execution counter
    """
    try:
        table = dynamodb.Table(SYSTEM_STATE_TABLE)
        response = table.get_item(Key={'state_key': 'execution_counter'})

        if 'Item' in response:
            current_counter = int(response['Item'].get('counter', 0))
        else:
            current_counter = 0

        # Increment counter for next execution
        new_counter = current_counter + 1
        table.put_item(
            Item={
                'state_key': 'execution_counter',
                'counter': new_counter,
                'updated_at': datetime.utcnow().isoformat()
            }
        )

        logger.info(f"Execution counter: {current_counter} (incremented to {new_counter})")
        return current_counter

    except Exception as e:
        logger.error(f"Error managing execution counter: {e}")
        return 0  # Default to 0 on error


def apply_tiered_analysis(
    symbols: List[str],
    execution_counter: int,
    tier1_size: int = 100,
    tier2_size: int = 200,
    tier3_size: int = 700
) -> List[str]:
    """
    Apply tiered analysis based on execution counter

    Tier 1 (top tier1_size by liquidity): Analyze every execution
    Tier 2 (next tier2_size): Analyze every 5th execution
    Tier 3 (next tier3_size): Analyze every 15th execution

    Args:
        symbols: List of symbols (assumed to be sorted by liquidity, highest first)
        execution_counter: Current execution counter
        tier1_size: Size of tier 1 (default 100)
        tier2_size: Size of tier 2 (default 200)
        tier3_size: Size of tier 3 (default 700)

    Returns:
        List of symbols to analyze this execution
    """
    # Split into tiers
    tier1 = symbols[:tier1_size]
    tier2 = symbols[tier1_size:tier1_size + tier2_size]
    tier3 = symbols[tier1_size + tier2_size:tier1_size + tier2_size + tier3_size]

    # Tier 1: Always analyze
    symbols_to_analyze = tier1.copy()

    # Tier 2: Every 5th execution
    if execution_counter % 5 == 0:
        symbols_to_analyze.extend(tier2)

    # Tier 3: Every 15th execution
    if execution_counter % 15 == 0:
        symbols_to_analyze.extend(tier3)

    logger.info(
        f"Tiered analysis (counter={execution_counter}): "
        f"Tier1={len(tier1)} (always), "
        f"Tier2={len(tier2)} (every 5th: {'YES' if execution_counter % 5 == 0 else 'NO'}), "
        f"Tier3={len(tier3)} (every 15th: {'YES' if execution_counter % 15 == 0 else 'NO'}) - "
        f"Total analyzing: {len(symbols_to_analyze)}"
    )

    return symbols_to_analyze


def lambda_handler(event, context):
    """
    Main Lambda handler for trading execution

    Args:
        event: Lambda event
        context: Lambda context

    Returns:
        Response with execution summary
    """
    logger.info("Trading executor started")

    try:
        # Get Alpaca credentials
        alpaca_client = get_alpaca_client()

        # Check if market is open
        if not alpaca_client.is_market_open():
            logger.info("Market is closed, skipping execution")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Market is closed'})
            }

        # Check system state (kill switch, circuit breaker)
        system_state = get_system_state()
        if not system_state.get('trading_enabled', True):
            logger.warning("Trading is disabled")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Trading disabled'})
            }

        if system_state.get('kill_switch', False):
            logger.warning("Kill switch is active")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Kill switch active'})
            }

        if system_state.get('circuit_breaker', False):
            logger.warning("Circuit breaker is active")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Circuit breaker active'})
            }

        # Get account info
        account = alpaca_client.get_account()
        portfolio_value = float(account.equity)
        buying_power = float(account.buying_power)
        cash_balance = float(account.cash)
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}, Buying power: ${buying_power:,.2f}, Cash: ${cash_balance:,.2f}")

        # Get current positions
        current_positions = get_current_positions(alpaca_client)

        # Reconcile positions (ensure DynamoDB matches Alpaca reality)
        positions_table = dynamodb.Table(POSITIONS_TABLE)
        reconciler = PositionReconciler(alpaca_client, positions_table)
        discrepancies = reconciler.reconcile()

        if discrepancies:
            # Log discrepancies at INFO level (routine state syncing, not errors)
            logger.info(f"Position reconciliation synced {len(discrepancies)} discrepancy(ies)")
            for disc in discrepancies:
                logger.info(f"  - {disc['type']}: {disc['symbol']} - {disc['details']}")
            # Note: Reconciliation automatically syncs state, so no alert needed
            # Only actual errors (failed DB writes) are logged as errors in PositionReconciler

        # Load trading configuration
        config = load_config()

        # Get use_margin setting from Parameter Store
        use_margin = get_use_margin_setting()

        # Initialize components with config
        market_scanner_config = config.get('market_scanner', {})
        strategies_config = config.get('strategies', {})
        risk_management_config = config.get('risk_management', {})
        trading_config = config.get('trading', {})

        market_scanner = MarketScanner(alpaca_client, config=market_scanner_config)
        strategy_manager = StrategyManager(
            consensus_threshold=trading_config.get('consensus_threshold', 0.20),
            config=strategies_config
        )
        risk_manager = RiskManager(
            config=risk_management_config,
            use_margin=use_margin
        )

        # Get blacklist
        blacklist = get_blacklist()

        # Initialize wash sale tracker
        tax_compliance_config = config.get('tax_compliance', {})
        wash_sale_enabled = tax_compliance_config.get('wash_sale_tracking_enabled', False)
        wash_sale_period = tax_compliance_config.get('wash_sale_period_days', 30)
        wash_sale_tracker = WashSaleTracker(enabled=wash_sale_enabled, wash_sale_period_days=wash_sale_period)

        # Get closed trades for wash sale tracking
        closed_trades = get_closed_trades(days_back=wash_sale_period) if wash_sale_enabled else []
        if wash_sale_enabled:
            logger.info(f"Wash sale tracking enabled: {len(closed_trades)} closed trades retrieved")

        # Initialize PDT tracker for growth mode
        pdt_config = config.get('pdt_tracking', {})
        pdt_enabled = pdt_config.get('enabled', True)
        pdt_threshold = float(pdt_config.get('threshold', 25000.0))
        pdt_limit = int(pdt_config.get('limit', 3))
        pdt_tracker = PDTTracker(pdt_threshold=pdt_threshold, pdt_limit=pdt_limit)

        # Get current PDT status (for EOD exit logic)
        pdt_check_result = None
        if pdt_enabled:
            # Get recent trades for PDT counting
            trades_table = dynamodb.Table(TRADES_TABLE)
            lookback_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            recent_trades_response = trades_table.scan(
                FilterExpression='#ts >= :lookback',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':lookback': lookback_date}
            )
            recent_trades_list = recent_trades_response.get('Items', [])

            # Check PDT status (use dummy symbol to get current count)
            pdt_check = pdt_tracker.check_pdt_compliance(
                symbol='_DUMMY',
                side='buy',
                account_value=portfolio_value,
                recent_trades=recent_trades_list,
                current_positions=current_positions
            )

            pdt_check_result = {
                'account_value': pdt_check.account_value,
                'day_trades_count': pdt_check.day_trades_count,
                'day_trades_limit': pdt_check.day_trades_limit,
                'pdt_threshold': pdt_check.pdt_threshold
            }

            logger.info(
                f"PDT Status: {pdt_check.day_trades_count}/{pdt_check.day_trades_limit} day trades, "
                f"Account: ${portfolio_value:,.2f} {'(PDT Exempt)' if portfolio_value >= pdt_threshold else '(PDT Restricted)'}"
            )

        # Check if deleveraging is needed (margin disabled but has margin debt)
        position_management_config = config.get('position_management', {})
        deleverage_result = deleverage_margin_if_needed(
            alpaca_client=alpaca_client,
            risk_manager=risk_manager,
            cash_balance=cash_balance,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            position_management_config=position_management_config
        )

        # If deleveraging occurred, update cash_balance and current_positions
        if deleverage_result['deleveraged']:
            logger.info(
                f"Deleveraging completed: sold {deleverage_result['positions_sold']} positions, "
                f"freed ${deleverage_result['cash_freed']:,.2f}"
            )
            # Update cash balance after deleveraging
            cash_balance += deleverage_result['cash_freed']
            logger.info(f"Updated cash balance after deleveraging: ${cash_balance:,.2f}")

            # Refresh current positions from Alpaca after deleveraging
            current_positions = get_current_positions(alpaca_client)
            logger.info(f"Refreshed positions after deleveraging: {len(current_positions)} positions remaining")

        # CHECK 1: End-of-day exit (FIRST - closes ALL positions if after 3:45 PM ET)
        # This takes priority over all other trading logic
        # Pass PDT status for growth mode logic
        eod_exit_result = check_end_of_day_exit(
            alpaca_client=alpaca_client,
            risk_manager=risk_manager,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            config=config,
            pdt_check_result=pdt_check_result
        )

        # If EOD exit triggered, all positions closed - skip normal trading
        if eod_exit_result['eod_exit_triggered']:
            logger.info("End-of-day exit triggered - all positions closed, skipping normal trading cycle")
            # Update balances based on margin setting
            if risk_manager.use_margin:
                buying_power += eod_exit_result['cash_freed']
            else:
                cash_balance += eod_exit_result['cash_freed']

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'End-of-day exit complete',
                    'positions_closed': eod_exit_result['positions_closed'],
                    'cash_freed': eod_exit_result['cash_freed'],
                    'total_pl': sum(d.get('realized_pl', 0) for d in eod_exit_result['details'] if d.get('action') == 'exited')
                })
            }

        # CHECK 2: Check stop-loss levels (including trailing stops and breakeven)
        # This ensures we exit losing positions early, protecting capital
        stop_loss_result = check_stop_losses(
            alpaca_client=alpaca_client,
            risk_manager=risk_manager,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            historical_data={},  # Will be populated if needed
            config=config
        )

        # If stop-losses were triggered, update cash_balance and refresh positions
        if stop_loss_result['stop_losses_triggered'] > 0:
            logger.info(
                f"Stop-loss enforcement: triggered {stop_loss_result['stop_losses_triggered']}, "
                f"sold {stop_loss_result['positions_sold']} positions, "
                f"freed ${stop_loss_result['cash_freed']:,.2f}"
            )
            # Update balances based on margin setting
            if risk_manager.use_margin:
                buying_power += stop_loss_result['cash_freed']
            else:
                cash_balance += stop_loss_result['cash_freed']
            logger.info(f"Updated balance after stop-loss sells: cash=${cash_balance:,.2f}, buying_power=${buying_power:,.2f}")

            # Refresh current positions from Alpaca after stop-loss sells
            current_positions = get_current_positions(alpaca_client)
            logger.info(f"Refreshed positions after stop-loss sells: {len(current_positions)} positions remaining")

        # CHECK 3: Check take-profit targets (partial exits on winning positions)
        # This locks in profits while letting remaining position run
        take_profit_result = check_take_profit_targets(
            alpaca_client=alpaca_client,
            risk_manager=risk_manager,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            config=config
        )

        # If take-profits were triggered, update cash_balance and refresh positions
        if take_profit_result['take_profits_triggered'] > 0:
            logger.info(
                f"Take-profit enforcement: triggered {take_profit_result['take_profits_triggered']}, "
                f"sold {take_profit_result['positions_sold']} partial positions, "
                f"freed ${take_profit_result['cash_freed']:,.2f}"
            )
            # Update balances based on margin setting
            if risk_manager.use_margin:
                buying_power += take_profit_result['cash_freed']
            else:
                cash_balance += take_profit_result['cash_freed']
            logger.info(f"Updated balance after take-profit sells: cash=${cash_balance:,.2f}, buying_power=${buying_power:,.2f}")

            # Refresh current positions from Alpaca after take-profit sells
            current_positions = get_current_positions(alpaca_client)
            logger.info(f"Refreshed positions after take-profit sells: {len(current_positions)} positions remaining")

        # Dynamically discover tradeable stocks using config values
        trading_universe = market_scanner.get_tradeable_universe(
            min_price=float(market_scanner_config.get('min_stock_price', '5.0')),
            max_price=float(market_scanner_config.get('max_stock_price', '1000.0')),
            min_volume=int(market_scanner_config.get('min_daily_volume', '1000000')),
            max_symbols=int(market_scanner_config.get('max_universe_size', '50')),
            exclude_symbols=blacklist,
        )
        logger.info(f"Trading universe (base): {len(trading_universe)} stocks discovered")

        # Scan for intraday volume spikes (catches momentum stocks breaking out TODAY)
        volume_spike_stocks = market_scanner.scan_intraday_volume_spikes(
            min_price=float(market_scanner_config.get('min_stock_price', '5.0')),
            max_price=float(market_scanner_config.get('max_stock_price', '1000.0')),
            exclude_symbols=blacklist,
        )

        # Combine base universe with volume spike stocks (remove duplicates)
        if volume_spike_stocks:
            original_size = len(trading_universe)
            trading_universe = list(set(trading_universe + volume_spike_stocks))
            logger.info(
                f"Added {len(volume_spike_stocks)} volume spike stocks "
                f"({len(trading_universe) - original_size} new) to universe"
            )

        logger.info(f"Trading universe (total): {len(trading_universe)} stocks")

        # Apply tiered analysis if enabled
        tiered_analysis_enabled = config.get('execution', {}).get('tiered_analysis', True)
        if tiered_analysis_enabled and len(trading_universe) > 100:
            execution_counter = get_and_increment_execution_counter()
            tier1_size = int(market_scanner_config.get('tier1_size', 100))
            tier2_size = int(market_scanner_config.get('tier2_size', 200))
            tier3_size = int(market_scanner_config.get('tier3_size', 700))

            trading_universe = apply_tiered_analysis(
                trading_universe,
                execution_counter,
                tier1_size=tier1_size,
                tier2_size=tier2_size,
                tier3_size=tier3_size
            )

        # Execute trading cycle
        execution_summary = execute_trading_cycle(
            alpaca_client,
            strategy_manager,
            risk_manager,
            market_scanner,
            trading_universe,
            portfolio_value,
            current_positions,
            system_state,
            buying_power,
            cash_balance,
            position_management_config,
            wash_sale_tracker,
            closed_trades,
            config  # Pass full config for PDT tracking
        )

        # Update risk metrics
        update_risk_metrics(alpaca_client, portfolio_value, current_positions)

        # Check if circuit breaker should be activated
        risk_metrics = get_latest_risk_metrics()
        if risk_manager.should_activate_circuit_breaker(risk_metrics):
            activate_circuit_breaker("Risk limits breached")

        # Log strategy health report periodically
        strategy_health = strategy_manager.get_health_report()
        logger.debug(f"Strategy health: {strategy_health}")

        logger.info(f"Trading executor completed: {execution_summary}")

        return {
            'statusCode': 200,
            'body': json.dumps(execution_summary)
        }

    except Exception as e:
        logger.error(f"Error in trading executor: {e}", exc_info=True)
        send_alert(f"Trading Executor Error: {str(e)}", "ERROR")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def get_alpaca_client() -> AlpacaClient:
    """Get Alpaca client with credentials from Secrets Manager"""
    try:
        secret = secrets_manager.get_secret_value(SecretId=ALPACA_SECRET_NAME)
        secret_data = json.loads(secret['SecretString'])

        api_key = secret_data['api_key']
        secret_key = secret_data['secret_key']
        paper = secret_data.get('paper', True)

        return AlpacaClient(api_key, secret_key, paper)

    except Exception as e:
        logger.error(f"Error getting Alpaca credentials: {e}")
        raise


def get_system_state() -> Dict:
    """Get system state from DynamoDB"""
    table = dynamodb.Table(SYSTEM_STATE_TABLE)

    state = {
        'trading_enabled': True,
        'kill_switch': False,
        'circuit_breaker': False,
    }

    for key in state.keys():
        try:
            response = table.get_item(Key={'state_key': key})
            if 'Item' in response:
                state[key] = response['Item']['value']
        except Exception as e:
            logger.warning(f"Error getting system state {key}: {e}")

    return state


def get_blacklist() -> List[str]:
    """Get blacklisted symbols from S3"""
    try:
        response = s3.get_object(Bucket=DATA_BUCKET, Key='blacklist.json')
        blacklist = json.loads(response['Body'].read())
        logger.info(f"Loaded blacklist: {len(blacklist)} symbols")
        return blacklist
    except s3.exceptions.NoSuchKey:
        logger.info("No blacklist found, using empty list")
        return []
    except Exception as e:
        logger.warning(f"Error loading blacklist: {e}")
        return []


def get_current_positions(alpaca_client: AlpacaClient) -> Dict:
    """Get current positions"""
    positions = alpaca_client.get_positions()

    position_dict = {}
    for pos in positions:
        position_dict[pos.symbol] = {
            'quantity': float(pos.qty),
            'avg_entry_price': float(pos.avg_entry_price),
            'current_price': float(pos.current_price),
            'market_value': float(pos.market_value),
            'unrealized_pl': float(pos.unrealized_pl),
        }

    return position_dict


def execute_trading_cycle(
    alpaca_client: AlpacaClient,
    strategy_manager: StrategyManager,
    risk_manager: RiskManager,
    market_scanner: MarketScanner,
    trading_universe: List[str],
    portfolio_value: float,
    current_positions: Dict,
    system_state: Dict,
    buying_power: float,
    cash_balance: float,
    position_management_config: Dict = None,
    wash_sale_tracker: WashSaleTracker = None,
    closed_trades: List[Dict] = None,
    config: Dict = None
) -> Dict:
    """Execute trading cycle for all symbols"""

    # Default to empty dict if config not provided
    if config is None:
        config = {}

    summary = {
        'symbols_analyzed': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'trades_executed': 0,
        'trades_rejected': 0,
        'errors': 0,
    }

    # Get historical data for all symbols (including SPY for benchmark and current positions for rebalancing)
    position_symbols = list(current_positions.keys()) if current_positions else []
    all_symbols = list(set(trading_universe + ['SPY'] + position_symbols))
    start_date = datetime.utcnow() - timedelta(days=300)  # ~300 days to ensure 200+ trading days
    end_date = datetime.utcnow()

    logger.info(f"Fetching historical data for {len(all_symbols)} symbols (universe: {len(trading_universe)}, positions: {len(position_symbols)})")
    historical_data = alpaca_client.get_historical_bars(all_symbols, start_date, end_date)

    # Get benchmark data (SPY)
    benchmark_data = historical_data.get('SPY')

    # Fetch 5-minute intraday bars for day trading strategies
    # Get today's market open (9:30 AM ET) in UTC
    from datetime import timezone
    now_utc = datetime.now(timezone.utc)
    # Convert to ET (approximate - UTC-5 for EST, UTC-4 for EDT)
    # For simplicity, use UTC-5 as default (adjust for DST in production)
    et_offset = timedelta(hours=5)
    now_et = now_utc - et_offset

    # Get today's market open time in UTC
    market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    intraday_start = market_open_et + et_offset  # Convert back to UTC
    intraday_end = now_utc

    # Only fetch intraday data if market is open (after 9:30 AM ET)
    if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30):
        logger.info("Market not yet open, skipping intraday data fetch")
        intraday_data = {}
    else:
        logger.info(f"Fetching 5-minute intraday bars for {len(all_symbols)} symbols from {intraday_start} to {intraday_end}")
        # Use cached intraday data to avoid redundant fetches
        cache_enabled = config.get('execution', {}).get('cache_intraday_data', True)
        intraday_data = get_cached_intraday_data(
            alpaca_client, all_symbols, intraday_start, intraday_end, cache_enabled=cache_enabled
        )

    # Get risk metrics
    risk_metrics = get_latest_risk_metrics()

    # Apply pre-filter to reduce analysis load
    pre_filter_enabled = config.get('market_scanner', {}).get('pre_filter_enabled', True)
    if pre_filter_enabled:
        filtered_universe, pre_filter_stats = market_scanner.pre_filter_candidates(
            trading_universe, historical_data, intraday_data
        )
        logger.info(f"Pre-filter reduced universe from {len(trading_universe)} to {len(filtered_universe)} symbols")
        summary['pre_filter_stats'] = pre_filter_stats
        trading_universe = filtered_universe

    # CRITICAL: Always include held positions in analysis, regardless of pre-filter
    # Held positions must be evaluated EVERY execution for strategy signals and exits
    if position_symbols:
        positions_not_in_universe = [s for s in position_symbols if s not in trading_universe]
        if positions_not_in_universe:
            trading_universe = list(set(trading_universe + position_symbols))
            logger.info(f"Added {len(positions_not_in_universe)} held positions to analysis universe (total: {len(trading_universe)})")
            logger.info(f"Held positions added: {positions_not_in_universe}")

    # Analyze each symbol
    for symbol in trading_universe:
        try:
            summary['symbols_analyzed'] += 1

            data = historical_data.get(symbol)
            if data is None or data.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Get intraday data for this symbol and SPY (for relative strength)
            symbol_intraday_data = intraday_data.get(symbol)
            spy_intraday_data = intraday_data.get('SPY')

            # Generate consensus signal (pass intraday_data for day trading strategies)
            action, consensus_score, strategy_signals = strategy_manager.generate_consensus_signal(
                symbol,
                data,
                benchmark_data=benchmark_data,
                intraday_data=symbol_intraday_data,
                spy_intraday_data=spy_intraday_data
            )

            if action == 'buy':
                summary['buy_signals'] += 1
                # Execute buy
                result = execute_buy(
                    alpaca_client,
                    risk_manager,
                    symbol,
                    data,
                    portfolio_value,
                    current_positions,
                    consensus_score,
                    strategy_signals,
                    risk_metrics,
                    system_state,
                    buying_power,
                    cash_balance,
                    historical_data,  # Pass all historical data for rebalancing
                    position_management_config,  # Pass config
                    wash_sale_tracker,  # Pass wash sale tracker
                    closed_trades,  # Pass closed trades for wash sale check
                    config  # Pass full config for PDT tracking
                )
                if result['executed']:
                    summary['trades_executed'] += 1
                    # Update balances after successful trade based on margin setting
                    trade_cost = data['close'].iloc[-1] * result.get('quantity', 0)
                    if risk_manager.use_margin:
                        # Margin enabled: update buying power
                        buying_power -= trade_cost
                    else:
                        # Margin disabled: update cash balance
                        cash_balance -= trade_cost
                else:
                    summary['trades_rejected'] += 1

            elif action == 'sell':
                summary['sell_signals'] += 1
                # Execute sell (if we have a position)
                if symbol in current_positions:
                    result = execute_sell(
                        alpaca_client,
                        risk_manager,
                        symbol,
                        data,
                        portfolio_value,
                        current_positions,
                        consensus_score,
                        strategy_signals,
                        risk_metrics,
                        system_state,
                        config  # Pass full config for PDT tracking
                    )
                    if result['executed']:
                        summary['trades_executed'] += 1
                        # Update balances after successful sell
                        sell_proceeds = data['close'].iloc[-1] * result.get('quantity', 0)
                        if risk_manager.use_margin:
                            buying_power += sell_proceeds
                        else:
                            cash_balance += sell_proceeds
                    else:
                        summary['trades_rejected'] += 1

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            summary['errors'] += 1

    return summary


def execute_buy(
    alpaca_client, risk_manager, symbol, data, portfolio_value,
    current_positions, consensus_score, strategy_signals, risk_metrics, system_state,
    buying_power, cash_balance, historical_data, position_management_config: Dict = None,
    wash_sale_tracker: WashSaleTracker = None, closed_trades: List[Dict] = None,
    config: Dict = None
) -> Dict:
    """Execute buy trade with intelligent position rebalancing and wash sale tracking"""

    # Default to empty dict if config not provided
    if config is None:
        config = {}

    # Check wash sale rule first (before any other checks)
    if wash_sale_tracker and closed_trades is not None:
        wash_check = wash_sale_tracker.check_wash_sale(symbol, 'buy', closed_trades)
        if not wash_check.can_buy:
            logger.warning(f"⚠ WASH SALE BLOCKED: {wash_check.blocked_reason}")
            # Create rejected trade record
            trade = Trade.create_new(
                symbol=symbol,
                side='buy',
                quantity=0,
                order_type='market',
                strategy_signals=strategy_signals,
                consensus_score=consensus_score,
                risk_checks_passed=False,
                risk_check_details={'wash_sale': wash_check.blocked_reason}
            )
            trade.mark_rejected(f"Wash sale rule: {wash_check.blocked_reason}")
            save_trade(trade)
            return {
                'executed': False,
                'reason': 'Wash sale rule violation',
                'details': wash_check.blocked_reason,
                'days_until_clear': wash_check.days_until_clear
            }

    # Calculate position size
    atr = TechnicalIndicators.atr(data['high'], data['low'], data['close']).iloc[-1]
    current_price = data['close'].iloc[-1]

    quantity = risk_manager.calculate_position_size(
        symbol, current_price, portfolio_value, atr
    )

    if quantity == 0:
        logger.info(f"Position size too small for {symbol}")
        return {'executed': False, 'reason': 'Position size too small'}

    trade_cost = quantity * current_price

    # First perform risk check to see if we have sufficient funds
    # This will check either cash_balance or buying_power based on use_margin config
    preliminary_risk_check = risk_manager.check_trade_risk(
        symbol=symbol,
        side='buy',
        quantity=quantity,
        price=current_price,
        portfolio_value=portfolio_value,
        current_positions=current_positions,
        market_data=data,
        risk_metrics=risk_metrics,
        system_state=system_state,
        buying_power=buying_power,
        cash_balance=cash_balance
    )

    # Check if insufficient funds is the ONLY issue (no cash, but otherwise valid trade)
    has_cash_violation = any('Insufficient' in v for v in preliminary_risk_check.violations)

    if has_cash_violation:
        logger.info(
            f"Insufficient cash for {symbol}: Need ${trade_cost:,.2f}, have ${cash_balance:,.2f}. "
            f"Evaluating position rebalancing..."
        )

        # Initialize position evaluator with config
        position_evaluator = PositionEvaluator(config=position_management_config)

        # Evaluate if we should rebalance positions
        rebalance_decision = position_evaluator.evaluate_rebalance_decision(
            new_buy_symbol=symbol,
            new_buy_consensus=consensus_score,
            new_buy_price=current_price,
            new_buy_quantity=quantity,
            current_positions=current_positions,
            historical_data=historical_data,
            cash_needed=trade_cost
        )

        logger.info(f"Rebalance decision: {rebalance_decision.reason}")

        if rebalance_decision.should_rebalance:
            # Execute sells for weak positions
            logger.info(
                f"REBALANCING: Selling {len(rebalance_decision.positions_to_sell)} weak position(s) "
                f"to buy stronger position {symbol}"
            )

            total_cash_freed = 0.0
            for sell_symbol in rebalance_decision.positions_to_sell:
                sell_position = current_positions[sell_symbol]
                sell_quantity = sell_position['quantity']
                sell_price = sell_position['current_price']

                # Execute sell order with enhanced handling
                logger.info(
                    f"Rebalancing sell: {sell_quantity} {sell_symbol} @ ${sell_price:.2f} "
                    f"(market value: ${sell_position['market_value']:,.2f})"
                )

                sell_result = alpaca_client.place_market_order_with_confirmation(
                    symbol=sell_symbol,
                    qty=sell_quantity,
                    side='sell',
                    expected_price=sell_price,
                    wait_for_fill=True,
                    max_wait_seconds=30
                )

                if sell_result.status == OrderStatus.FILLED:
                    actual_cash_freed = sell_result.filled_quantity * sell_result.filled_avg_price
                    total_cash_freed += actual_cash_freed
                    logger.info(
                        f"Rebalance sell successful: {sell_result.filled_quantity} {sell_symbol} @ "
                        f"${sell_result.filled_avg_price:.2f}, freed ${actual_cash_freed:,.2f}"
                    )

                    # Track realized loss for wash sale (if applicable)
                    realized_pl = (sell_result.filled_avg_price - sell_position['avg_entry_price']) * sell_result.filled_quantity
                    # TODO: Fix wash sale tracker API mismatch - disabled for now (paper trading has no tax implications)
                    # if realized_pl < 0 and REALIZED_LOSSES_TABLE:
                    #     # Track the loss for wash sale rules
                    #     wash_tracker = WashSaleTracker(dynamodb.Table(REALIZED_LOSSES_TABLE), ENVIRONMENT, STAGE)
                    #     wash_tracker.record_loss(sell_symbol, abs(realized_pl), sell_result.filled_at or datetime.utcnow().isoformat())

                    # Record day trade for PDT if applicable (if we bought today and selling today)
                    if config.get('pdt_tracking', {}).get('enabled', True) and DAY_TRADES_TABLE:
                        try:
                            # Check if this sell completed a day trade
                            pdt_threshold = config.get('pdt_tracking', {}).get('threshold', 25000.0)
                            pdt_limit = config.get('pdt_tracking', {}).get('limit', 3)
                            pdt_tracker = PDTTracker(pdt_threshold=pdt_threshold, pdt_limit=pdt_limit)

                            # Get recent trades to check if this completed a day trade
                            trades_table = dynamodb.Table(TRADES_TABLE)
                            cutoff_date = (datetime.utcnow() - timedelta(days=1)).isoformat()  # Just today
                            recent_trades_response = trades_table.scan(
                                FilterExpression='#ts > :cutoff AND #status = :status AND symbol = :symbol',
                                ExpressionAttributeNames={
                                    '#ts': 'timestamp',
                                    '#status': 'status'
                                },
                                ExpressionAttributeValues={
                                    ':cutoff': cutoff_date,
                                    ':status': 'filled',
                                    ':symbol': sell_symbol
                                }
                            )
                            recent_symbol_trades = recent_trades_response.get('Items', [])

                            # Check if this sell completed a day trade (bought earlier today)
                            if pdt_tracker._would_create_day_trade(sell_symbol, 'sell', recent_symbol_trades, current_positions):
                                # Record to day-trades table
                                day_trades_table = dynamodb.Table(DAY_TRADES_TABLE)
                                trade_date = datetime.utcnow().date().isoformat()
                                account = alpaca_client.get_account()

                                day_trades_table.put_item(
                                    Item={
                                        'date': trade_date,
                                        'symbol': sell_symbol,
                                        'side': 'sell',
                                        'timestamp': sell_result.filled_at or datetime.utcnow().isoformat(),
                                        'account_value': float(account.equity),
                                        'ttl': int((datetime.utcnow() + timedelta(days=30)).timestamp())
                                    }
                                )
                                logger.info(f"✓ Recorded day trade for {sell_symbol} on {trade_date}")
                        except Exception as e:
                            logger.error(f"Error recording day trade for sell: {e}")

                    # Remove from current positions (so risk checks don't count it)
                    current_positions.pop(sell_symbol, None)

                    # Create trade record for the rebalancing sell
                    sell_trade = Trade.create_new(
                        symbol=sell_symbol,
                        side='sell',
                        quantity=sell_quantity,
                        order_type='market',
                        strategy_signals={'rebalancing': True},
                        consensus_score=0.0,  # This is a forced sell
                        risk_checks_passed=True,
                        risk_check_details={'reason': 'Position rebalancing'}
                    )
                    sell_trade.mark_filled(
                        sell_result.filled_avg_price,
                        sell_result.filled_quantity,
                        sell_result.order_id
                    )
                    sell_trade.calculate_realized_pl(sell_position['avg_entry_price'] * sell_result.filled_quantity)
                    save_trade(sell_trade)

                elif sell_result.status == OrderStatus.PARTIALLY_FILLED:
                    # Handle partial fill
                    actual_cash_freed = sell_result.filled_quantity * sell_result.filled_avg_price
                    total_cash_freed += actual_cash_freed
                    logger.warning(
                        f"Rebalance sell PARTIALLY filled: {sell_result.filled_quantity}/{sell_result.requested_quantity} "
                        f"{sell_symbol} @ ${sell_result.filled_avg_price:.2f}, freed ${actual_cash_freed:,.2f}"
                    )

                    # Still create trade record for what was filled
                    sell_trade = Trade.create_new(
                        symbol=sell_symbol,
                        side='sell',
                        quantity=sell_quantity,
                        order_type='market',
                        strategy_signals={'rebalancing': True},
                        consensus_score=0.0,
                        risk_checks_passed=True,
                        risk_check_details={'reason': 'Position rebalancing (partial fill)'}
                    )
                    sell_trade.mark_filled(
                        sell_result.filled_avg_price,
                        sell_result.filled_quantity,
                        sell_result.order_id
                    )
                    save_trade(sell_trade)

                else:
                    # Rejected or failed
                    logger.error(
                        f"Failed to execute rebalancing sell for {sell_symbol}: "
                        f"{sell_result.rejection_reason} - {sell_result.error_message}"
                    )

                    # Handle stock halt specifically
                    if sell_result.rejection_reason == RejectionReason.STOCK_HALTED:
                        logger.warning(f"{sell_symbol} is halted, cannot sell for rebalancing")
                        # Could add to temporary blacklist here

            # Update cash balance with freed cash
            cash_balance += total_cash_freed
            logger.info(
                f"Rebalancing complete. Cash freed: ${total_cash_freed:,.2f}. "
                f"New cash balance: ${cash_balance:,.2f}"
            )

            # Alert about rebalancing
            send_alert(
                f"REBALANCING: Sold {len(rebalance_decision.positions_to_sell)} positions "
                f"({', '.join(rebalance_decision.positions_to_sell)}) to buy {symbol}. "
                f"Cash freed: ${total_cash_freed:,.2f}",
                "INFO"
            )
        else:
            # Cannot rebalance, reject the buy
            logger.info(f"Cannot rebalance for {symbol}: {rebalance_decision.reason}")
            return {
                'executed': False,
                'reason': 'Insufficient cash and cannot rebalance',
                'details': rebalance_decision.reason
            }

    # Now perform final risk check with updated balances
    # (If we rebalanced, cash_balance and current_positions have changed)
    risk_check = risk_manager.check_trade_risk(
        symbol=symbol,
        side='buy',
        quantity=quantity,
        price=current_price,
        portfolio_value=portfolio_value,
        current_positions=current_positions,
        market_data=data,
        risk_metrics=risk_metrics,
        system_state=system_state,
        buying_power=buying_power,
        cash_balance=cash_balance
    )

    # Create trade record
    trade = Trade.create_new(
        symbol=symbol,
        side='buy',
        quantity=quantity,
        order_type='market',
        strategy_signals=strategy_signals,
        consensus_score=consensus_score,
        risk_checks_passed=risk_check.passed,
        risk_check_details={
            'violations': risk_check.violations,
            'warnings': risk_check.warnings,
            'risk_score': risk_check.risk_score,
        }
    )

    # Save trade to DynamoDB
    save_trade(trade)

    if not risk_check.passed:
        logger.warning(f"Risk check failed for {symbol}: {risk_check.violations}")
        trade.mark_rejected(f"Risk check failed: {', '.join(risk_check.violations)}")
        save_trade(trade)
        # Rejected trades are consolidated in daily report - no individual alert needed
        return {'executed': False, 'reason': 'Risk check failed', 'violations': risk_check.violations}

    # Check wash sale rule (only if we have the table configured)
    # TODO: Fix wash sale tracker API mismatch - disabled for now (paper trading has no tax implications)
    # if REALIZED_LOSSES_TABLE:
    #     wash_tracker = WashSaleTracker(dynamodb.Table(REALIZED_LOSSES_TABLE), ENVIRONMENT, STAGE)
    #     if wash_tracker.is_wash_sale_violation(symbol):
    #         logger.warning(f"Wash sale violation: Cannot buy {symbol} within 30 days of realizing loss")
    #         trade.mark_rejected("Wash sale rule violation (30-day window)")
    #         save_trade(trade)
    #         return {'executed': False, 'reason': 'Wash sale rule violation'}

    # Check PDT rule (Pattern Day Trading)
    if config.get('pdt_tracking', {}).get('enabled', True):
        # Get PDT configuration
        pdt_threshold = config.get('pdt_tracking', {}).get('threshold', 25000.0)
        pdt_limit = config.get('pdt_tracking', {}).get('limit', 3)

        # Initialize PDT tracker
        pdt_tracker = PDTTracker(pdt_threshold=pdt_threshold, pdt_limit=pdt_limit)

        # Get account value
        account = alpaca_client.get_account()
        account_value = float(account.equity)

        # Get recent trades for PDT calculation (last 7 days)
        trades_table = dynamodb.Table(TRADES_TABLE)
        cutoff_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent_trades_response = trades_table.scan(
            FilterExpression='#ts > :cutoff AND #status = :status',
            ExpressionAttributeNames={
                '#ts': 'timestamp',
                '#status': 'status'
            },
            ExpressionAttributeValues={
                ':cutoff': cutoff_date,
                ':status': 'filled'
            }
        )
        recent_trades_list = recent_trades_response.get('Items', [])

        # Check PDT compliance
        pdt_check = pdt_tracker.check_pdt_compliance(
            symbol=symbol,
            side='buy',
            account_value=account_value,
            recent_trades=recent_trades_list,
            current_positions=current_positions
        )

        # Log the check result
        if pdt_check.warning_message:
            logger.warning(f"PDT Warning for {symbol}: {pdt_check.warning_message}")

        # Block trade if it would violate PDT
        if not pdt_check.can_trade:
            logger.warning(f"PDT VIOLATION BLOCKED: {pdt_check.blocking_message}")
            trade.mark_rejected(f"PDT rule violation: {pdt_check.blocking_message}")
            save_trade(trade)

            # Send critical alert for PDT violations
            send_alert(
                subject=f"🚫 PDT Violation Blocked - {symbol}",
                message=f"PDT Rule Violation Prevented\n\n{pdt_check.blocking_message}",
                level="ERROR"
            )

            return {'executed': False, 'reason': 'PDT rule violation'}

    # Execute trade with enhanced order handling
    order_result = alpaca_client.place_market_order_with_confirmation(
        symbol=symbol,
        qty=quantity,
        side='buy',
        expected_price=current_price,
        wait_for_fill=True,
        max_wait_seconds=30
    )

    if order_result.status == OrderStatus.FILLED:
        # Check slippage tolerance
        slippage_tolerance = config.get('trading', {}).get('slippage_tolerance_percentage', 1.5)
        if abs(order_result.slippage_percentage) > slippage_tolerance:
            # Slippage exceeded tolerance - reject trade and cancel order if possible
            logger.error(
                f"EXCESSIVE SLIPPAGE on {symbol}: {order_result.slippage_percentage:+.2f}% exceeds tolerance of {slippage_tolerance}%. "
                f"Expected ${current_price:.2f}, filled @ ${order_result.filled_avg_price:.2f}. "
                f"Attempting to reverse trade..."
            )

            # Mark trade as rejected due to slippage
            trade.mark_rejected(f"Excessive slippage: {order_result.slippage_percentage:+.2f}% (tolerance: {slippage_tolerance}%)")
            trade.slippage_dollars = order_result.slippage_dollars
            trade.slippage_percentage = order_result.slippage_percentage
            trade.expected_price = current_price
            save_trade(trade)

            # Try to immediately reverse the trade by selling
            try:
                reverse_result = alpaca_client.place_market_order_with_confirmation(
                    symbol=symbol,
                    qty=order_result.filled_quantity,
                    side='sell',
                    expected_price=order_result.filled_avg_price,
                    wait_for_fill=True,
                    max_wait_seconds=30
                )

                if reverse_result.status == OrderStatus.FILLED:
                    logger.warning(
                        f"Successfully reversed excessive slippage trade for {symbol}. "
                        f"Sold @ ${reverse_result.filled_avg_price:.2f}"
                    )
                else:
                    logger.error(
                        f"Failed to reverse excessive slippage trade for {symbol}. "
                        f"Manual intervention may be required. Order ID: {order_result.order_id}"
                    )
                    # Send critical alert
                    send_alert(
                        "CRITICAL",
                        "Slippage Reversal Failed",
                        f"Failed to reverse excessive slippage buy for {symbol}. "
                        f"Position remains open. Manual review required."
                    )
            except Exception as e:
                logger.error(f"Error reversing excessive slippage trade for {symbol}: {e}")
                send_alert(
                    "CRITICAL",
                    "Slippage Reversal Error",
                    f"Error reversing excessive slippage buy for {symbol}: {e}. "
                    f"Manual review required."
                )

            return {
                'executed': False,
                'reason': 'Excessive slippage',
                'slippage_percentage': order_result.slippage_percentage,
                'tolerance': slippage_tolerance
            }

        # Order fully filled with acceptable slippage
        trade.mark_filled(
            order_result.filled_avg_price,
            order_result.filled_quantity,
            order_result.order_id,
            slippage_dollars=order_result.slippage_dollars,
            slippage_percentage=order_result.slippage_percentage
        )
        trade.expected_price = current_price
        save_trade(trade)

        # Log slippage if significant (but within tolerance)
        if abs(order_result.slippage_percentage) > 0.5:
            logger.warning(
                f"Significant slippage on {symbol}: expected ${current_price:.2f}, "
                f"filled @ ${order_result.filled_avg_price:.2f} ({order_result.slippage_percentage:+.2f}%)"
            )

        # Calculate and store stop-loss price
        stop_loss_price = order_result.filled_avg_price - (risk_manager.atr_stop_multiplier * atr)
        logger.info(
            f"Stop-loss set for {symbol}: ${stop_loss_price:.2f} "
            f"(entry: ${order_result.filled_avg_price:.2f}, ATR: ${atr:.2f}, multiplier: {risk_manager.atr_stop_multiplier}x)"
        )

        # Store stop-loss in DynamoDB position record
        try:
            from decimal import Decimal
            positions_table = dynamodb.Table(POSITIONS_TABLE)

            # Update or create position with stop-loss and peak_price tracking
            positions_table.put_item(
                Item={
                    'symbol': symbol,
                    'quantity': Decimal(str(order_result.filled_quantity)),
                    'avg_entry_price': Decimal(str(order_result.filled_avg_price)),
                    'stop_loss_price': Decimal(str(stop_loss_price)),
                    'initial_stop_loss': Decimal(str(stop_loss_price)),  # Preserve original stop-loss
                    'trailing_stop_price': Decimal(str(stop_loss_price)),  # Initialize trailing stop to initial stop
                    'breakeven_stop_activated': False,  # Breakeven not yet activated
                    'peak_price': Decimal(str(order_result.filled_avg_price)),  # Initialize peak to entry price
                    'trailing_stop_active': False,  # Will be activated after take-profit
                    'take_profit_level': 0,  # Progressive take-profit: starts at level 0
                    'entry_timestamp': order_result.filled_at or datetime.utcnow().isoformat(),
                    'last_updated': datetime.utcnow().isoformat(),
                }
            )
            logger.info(f"✓ Stored position with stop-loss for {symbol} in DynamoDB")
        except Exception as e:
            logger.error(f"Error storing stop-loss for {symbol}: {e}")
            # Don't fail the trade if DynamoDB update fails

        # Record day trade if applicable (for tracking purposes)
        if config.get('pdt_tracking', {}).get('enabled', True) and DAY_TRADES_TABLE:
            try:
                # Check if this was a day trade
                pdt_threshold = config.get('pdt_tracking', {}).get('threshold', 25000.0)
                pdt_limit = config.get('pdt_tracking', {}).get('limit', 3)
                pdt_tracker = PDTTracker(pdt_threshold=pdt_threshold, pdt_limit=pdt_limit)

                # Get recent trades to check if this completed a day trade
                trades_table = dynamodb.Table(TRADES_TABLE)
                cutoff_date = (datetime.utcnow() - timedelta(days=1)).isoformat()  # Just today
                recent_trades_response = trades_table.scan(
                    FilterExpression='#ts > :cutoff AND #status = :status AND symbol = :symbol',
                    ExpressionAttributeNames={
                        '#ts': 'timestamp',
                        '#status': 'status'
                    },
                    ExpressionAttributeValues={
                        ':cutoff': cutoff_date,
                        ':status': 'filled',
                        ':symbol': symbol
                    }
                )
                recent_symbol_trades = recent_trades_response.get('Items', [])

                # Check if this buy completed a day trade (sold earlier today)
                if pdt_tracker._would_create_day_trade(symbol, 'buy', recent_symbol_trades, current_positions):
                    # Record to day-trades table
                    day_trades_table = dynamodb.Table(DAY_TRADES_TABLE)
                    trade_date = datetime.utcnow().date().isoformat()
                    account = alpaca_client.get_account()

                    day_trades_table.put_item(
                        Item={
                            'date': trade_date,
                            'symbol': symbol,
                            'side': 'buy',
                            'timestamp': order_result.filled_at or datetime.utcnow().isoformat(),
                            'account_value': float(account.equity),
                            'ttl': int((datetime.utcnow() + timedelta(days=30)).timestamp())
                        }
                    )
                    logger.info(f"✓ Recorded day trade for {symbol} on {trade_date}")
            except Exception as e:
                logger.error(f"Error recording day trade: {e}")

        logger.info(
            f"Buy order FILLED: {order_result.filled_quantity} {symbol} @ ${order_result.filled_avg_price:.2f} "
            f"(slippage: {order_result.slippage_percentage:+.2f}%)"
        )
        return {
            'executed': True,
            'order_id': order_result.order_id,
            'quantity': order_result.filled_quantity,
            'price': order_result.filled_avg_price,
            'slippage': order_result.slippage_percentage
        }

    elif order_result.status == OrderStatus.PARTIALLY_FILLED:
        # Partial fill - less quantity than requested
        trade.mark_filled(
            order_result.filled_avg_price,
            order_result.filled_quantity,
            order_result.order_id,
            slippage_dollars=order_result.slippage_dollars,
            slippage_percentage=order_result.slippage_percentage
        )
        trade.expected_price = current_price
        save_trade(trade)

        logger.warning(
            f"Buy order PARTIALLY filled: {order_result.filled_quantity}/{order_result.requested_quantity} "
            f"{symbol} @ ${order_result.filled_avg_price:.2f}"
        )

        # Calculate and store stop-loss price for partial fill
        stop_loss_price = order_result.filled_avg_price - (risk_manager.atr_stop_multiplier * atr)
        logger.info(f"Stop-loss set for {symbol} (partial fill): ${stop_loss_price:.2f}")

        try:
            from decimal import Decimal
            positions_table = dynamodb.Table(POSITIONS_TABLE)
            positions_table.put_item(
                Item={
                    'symbol': symbol,
                    'quantity': Decimal(str(order_result.filled_quantity)),
                    'avg_entry_price': Decimal(str(order_result.filled_avg_price)),
                    'stop_loss_price': Decimal(str(stop_loss_price)),
                    'initial_stop_loss': Decimal(str(stop_loss_price)),  # Preserve original stop-loss
                    'trailing_stop_price': Decimal(str(stop_loss_price)),  # Initialize trailing stop to initial stop
                    'breakeven_stop_activated': False,  # Breakeven not yet activated
                    'peak_price': Decimal(str(order_result.filled_avg_price)),  # Initialize peak to entry price
                    'trailing_stop_active': False,  # Will be activated after take-profit
                    'take_profit_level': 0,  # Progressive take-profit: starts at level 0
                    'entry_timestamp': order_result.filled_at or datetime.utcnow().isoformat(),
                    'last_updated': datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Error storing stop-loss for partial fill {symbol}: {e}")

        # Partial fills are consolidated in daily report - no individual alert needed
        return {
            'executed': True,
            'order_id': order_result.order_id,
            'quantity': order_result.filled_quantity,
            'price': order_result.filled_avg_price,
            'partial_fill': True
        }

    elif order_result.status == OrderStatus.REJECTED:
        # Order rejected by broker
        rejection_reason = f"{order_result.rejection_reason.value}: {order_result.error_message}"
        trade.mark_rejected(rejection_reason)
        save_trade(trade)

        logger.error(f"Buy order REJECTED for {symbol}: {rejection_reason}")
        # Rejected trades are consolidated in daily report - no individual alert needed

        return {
            'executed': False,
            'reason': 'Order rejected',
            'rejection_reason': order_result.rejection_reason.value,
            'error': order_result.error_message
        }

    else:
        # Failed or timeout
        trade.mark_rejected(f"Order failed: {order_result.error_message or 'Unknown error'}")
        save_trade(trade)
        logger.error(f"Buy order FAILED for {symbol}: {order_result.error_message}")
        return {
            'executed': False,
            'reason': 'Order execution failed',
            'error': order_result.error_message
        }


def execute_sell(
    alpaca_client, risk_manager, symbol, data, portfolio_value,
    current_positions, consensus_score, strategy_signals, risk_metrics, system_state,
    config: Dict = None
) -> Dict:
    """Execute sell trade"""

    # Default to empty dict if config not provided
    if config is None:
        config = {}

    position = current_positions[symbol]
    quantity = position['quantity']
    current_price = data['close'].iloc[-1]

    # Risk check
    risk_check = risk_manager.check_trade_risk(
        symbol=symbol,
        side='sell',
        quantity=quantity,
        price=current_price,
        portfolio_value=portfolio_value,
        current_positions=current_positions,
        market_data=data,
        risk_metrics=risk_metrics,
        system_state=system_state
    )

    # Create trade record
    trade = Trade.create_new(
        symbol=symbol,
        side='sell',
        quantity=quantity,
        order_type='market',
        strategy_signals=strategy_signals,
        consensus_score=consensus_score,
        risk_checks_passed=risk_check.passed,
        risk_check_details={
            'violations': risk_check.violations,
            'warnings': risk_check.warnings,
            'risk_score': risk_check.risk_score,
        }
    )

    # Save trade
    save_trade(trade)

    # Execute trade with enhanced order handling (sell signals are higher priority, so we execute even with warnings)
    order_result = alpaca_client.place_market_order_with_confirmation(
        symbol=symbol,
        qty=quantity,
        side='sell',
        expected_price=current_price,
        wait_for_fill=True,
        max_wait_seconds=30
    )

    if order_result.status == OrderStatus.FILLED:
        # Check slippage tolerance (but more lenient on sells since we want to exit)
        slippage_tolerance = config.get('trading', {}).get('slippage_tolerance_percentage', 1.5)
        # For sells, we allow 1.5x the normal tolerance (exiting is more important)
        sell_slippage_tolerance = slippage_tolerance * 1.5

        if abs(order_result.slippage_percentage) > sell_slippage_tolerance:
            # Excessive slippage on sell - log warning but don't reverse (we wanted out)
            logger.error(
                f"EXCESSIVE SLIPPAGE on {symbol} sell: {order_result.slippage_percentage:+.2f}% exceeds tolerance of {sell_slippage_tolerance:.1f}%. "
                f"Expected ${current_price:.2f}, filled @ ${order_result.filled_avg_price:.2f}. "
                f"Trade completed (exiting position takes priority)."
            )
            send_alert(
                "WARNING",
                "Excessive Sell Slippage",
                f"Excessive slippage on {symbol} sell: {order_result.slippage_percentage:+.2f}% "
                f"(tolerance: {sell_slippage_tolerance:.1f}%). Trade completed but review pricing."
            )

        # Order fully filled
        trade.mark_filled(
            order_result.filled_avg_price,
            order_result.filled_quantity,
            order_result.order_id,
            slippage_dollars=order_result.slippage_dollars,
            slippage_percentage=order_result.slippage_percentage
        )
        trade.expected_price = current_price
        trade.calculate_realized_pl(position['avg_entry_price'] * order_result.filled_quantity)
        save_trade(trade)

        # Track realized loss for wash sale tracking
        realized_pl = (order_result.filled_avg_price - position['avg_entry_price']) * order_result.filled_quantity
        # TODO: Fix wash sale tracker API mismatch - disabled for now (paper trading has no tax implications)
        # if realized_pl < 0 and REALIZED_LOSSES_TABLE:
        #     wash_tracker = WashSaleTracker(dynamodb.Table(REALIZED_LOSSES_TABLE), ENVIRONMENT, STAGE)
        #     wash_tracker.record_loss(
        #         symbol,
        #         abs(realized_pl),
        #         order_result.filled_at or datetime.utcnow().isoformat()
        #     )
        #     logger.info(f"Recorded realized loss for {symbol}: ${abs(realized_pl):.2f} (wash sale tracking)")

        # Check if this creates a day trade (bought and sold same day)
        # TODO: Fix PDT tracker API mismatch - disabled for now (paper trading has unlimited day trades)
        # if DAY_TRADES_TABLE:
        #     pdt_tracker = PDTTracker(dynamodb.Table(DAY_TRADES_TABLE), ENVIRONMENT, STAGE)
        #     if pdt_tracker.would_create_day_trade(symbol, 'sell'):
        #         pdt_tracker.record_day_trade(symbol, order_result.filled_at or datetime.utcnow().isoformat())
        #         logger.info(f"Recorded day trade for {symbol}")

        # Log slippage if significant (but within tolerance)
        if abs(order_result.slippage_percentage) > 0.5 and abs(order_result.slippage_percentage) <= sell_slippage_tolerance:
            logger.warning(
                f"Significant slippage on {symbol} sell: expected ${current_price:.2f}, "
                f"filled @ ${order_result.filled_avg_price:.2f} ({order_result.slippage_percentage:+.2f}%)"
            )

        logger.info(
            f"Sell order FILLED: {order_result.filled_quantity} {symbol} @ ${order_result.filled_avg_price:.2f} "
            f"(P&L: ${realized_pl:+.2f}, slippage: {order_result.slippage_percentage:+.2f}%)"
        )
        return {
            'executed': True,
            'order_id': order_result.order_id,
            'quantity': order_result.filled_quantity,
            'price': order_result.filled_avg_price,
            'realized_pl': realized_pl,
            'slippage': order_result.slippage_percentage
        }

    elif order_result.status == OrderStatus.PARTIALLY_FILLED:
        # Partial fill
        trade.mark_filled(order_result.filled_avg_price, order_result.filled_quantity, order_result.order_id)
        realized_pl = (order_result.filled_avg_price - position['avg_entry_price']) * order_result.filled_quantity
        trade.calculate_realized_pl(position['avg_entry_price'] * order_result.filled_quantity)
        save_trade(trade)

        logger.warning(
            f"Sell order PARTIALLY filled: {order_result.filled_quantity}/{order_result.requested_quantity} "
            f"{symbol} @ ${order_result.filled_avg_price:.2f}"
        )
        # Partial fills are consolidated in daily report - no individual alert needed
        return {
            'executed': True,
            'order_id': order_result.order_id,
            'quantity': order_result.filled_quantity,
            'price': order_result.filled_avg_price,
            'realized_pl': realized_pl,
            'partial_fill': True
        }

    elif order_result.status == OrderStatus.REJECTED:
        # Order rejected
        rejection_reason = f"{order_result.rejection_reason.value}: {order_result.error_message}"
        trade.mark_rejected(rejection_reason)
        save_trade(trade)

        logger.error(f"Sell order REJECTED for {symbol}: {rejection_reason}")
        # Rejected trades are consolidated in daily report - no individual alert needed
        # Note: Even critical rejections (stock halted, not tradeable) will be visible in daily report

        return {
            'executed': False,
            'reason': 'Order rejected',
            'rejection_reason': order_result.rejection_reason.value,
            'error': order_result.error_message
        }

    else:
        # Failed or timeout
        trade.mark_rejected(f"Order failed: {order_result.error_message or 'Unknown error'}")
        save_trade(trade)
        logger.error(f"Sell order FAILED for {symbol}: {order_result.error_message}")
        return {
            'executed': False,
            'reason': 'Order execution failed',
            'error': order_result.error_message
        }


def deleverage_margin_if_needed(
    alpaca_client,
    risk_manager,
    cash_balance: float,
    current_positions: Dict,
    portfolio_value: float,
    position_management_config: Dict
) -> Dict:
    """
    Check if margin is disabled but we have a margin debt (negative cash).
    If so, automatically sell the weakest positions to cover the debt.

    Returns dict with:
        - deleveraged: bool
        - positions_sold: int
        - cash_freed: float
        - symbols_sold: list
    """
    result = {
        'deleveraged': False,
        'positions_sold': 0,
        'cash_freed': 0.0,
        'symbols_sold': []
    }

    # Only deleverage if margin is disabled
    if risk_manager.use_margin:
        return result

    # Check if we have margin debt (negative cash)
    if cash_balance >= 0:
        logger.info(f"No margin debt detected. Cash balance: ${cash_balance:,.2f}")
        return result

    margin_debt = abs(cash_balance)
    logger.warning(
        f"MARGIN DELEVERAGING REQUIRED: use_margin=false but margin debt of ${margin_debt:,.2f} detected. "
        f"Will sell weakest positions to cover debt."
    )

    # Send alert about deleveraging
    send_alert(
        f"🚨 MARGIN DELEVERAGING INITIATED\n\n"
        f"Margin debt: ${margin_debt:,.2f}\n"
        f"Config: use_margin=false\n\n"
        f"Selling weakest positions to cover debt...",
        "CRITICAL"
    )

    # If no positions, we can't deleverage by selling
    if not current_positions:
        logger.error("Cannot deleverage: No positions to sell but margin debt exists!")
        send_alert(
            f"⚠️ DELEVERAGING FAILED\n\n"
            f"Margin debt: ${margin_debt:,.2f}\n"
            f"No positions available to sell!\n"
            f"Manual intervention required.",
            "CRITICAL"
        )
        return result

    # Fetch historical data for all positions (need it for scoring)
    historical_data = {}
    for symbol in current_positions.keys():
        try:
            data = alpaca_client.get_historical_data(
                symbol=symbol,
                timeframe='1Day',
                start=(datetime.now() - timedelta(days=300)).isoformat(),
                end=datetime.now().isoformat()
            )
            if data is not None and not data.empty:
                historical_data[symbol] = data
            else:
                logger.warning(f"No historical data for {symbol}, will use fallback scoring")
        except Exception as e:
            logger.warning(f"Error fetching data for {symbol}: {e}")

    # Initialize position evaluator
    evaluator = PositionEvaluator(config=position_management_config)

    # Score all positions (lower score = weaker position)
    scored_positions = []
    for symbol, position in current_positions.items():
        try:
            # Calculate position score
            score = evaluator._score_position(
                symbol=symbol,
                position=position,
                historical_data=historical_data.get(symbol),
                new_buy_consensus=0.0  # Not relevant for deleveraging
            )
            scored_positions.append({
                'symbol': symbol,
                'score': score,
                'quantity': position['quantity'],
                'current_price': position.get('current_price', 0.0),
                'market_value': position.get('market_value', 0.0)
            })
        except Exception as e:
            logger.error(f"Error scoring position {symbol}: {e}")
            # Assign low score if we can't calculate
            scored_positions.append({
                'symbol': symbol,
                'score': -100.0,  # Very low score = will be sold first
                'quantity': position['quantity'],
                'current_price': position.get('current_price', 0.0),
                'market_value': position.get('market_value', 0.0)
            })

    # Sort by score (lowest first = weakest positions)
    scored_positions.sort(key=lambda x: x['score'])

    logger.info(f"Scored {len(scored_positions)} positions for deleveraging:")
    for sp in scored_positions:
        logger.info(f"  {sp['symbol']}: score={sp['score']:.2f}, value=${sp['market_value']:,.2f}")

    # Sell positions starting from weakest until debt is covered
    cash_needed = margin_debt
    cash_freed_total = 0.0
    positions_sold = 0
    symbols_sold = []

    for position in scored_positions:
        if cash_freed_total >= cash_needed:
            break

        symbol = position['symbol']
        quantity = position['quantity']
        current_price = position['current_price']

        logger.info(
            f"Deleveraging: Selling {symbol} ({quantity} shares @ ${current_price:.2f}) "
            f"to cover margin debt. Position score: {position['score']:.2f}"
        )

        try:
            # Execute market sell order
            order_result = alpaca_client.place_market_order_with_confirmation(
                symbol=symbol,
                qty=quantity,
                side='sell',
                expected_price=current_price,
                wait_for_fill=True,
                max_wait_seconds=30
            )

            if order_result.status == OrderStatus.FILLED:
                # Calculate cash freed
                cash_freed = order_result.filled_avg_price * order_result.filled_quantity
                cash_freed_total += cash_freed
                positions_sold += 1
                symbols_sold.append(symbol)

                # Create and save trade record
                trade = Trade.create_new(
                    symbol=symbol,
                    side='sell',
                    quantity=order_result.filled_quantity,
                    order_type='market',
                    strategy_signals={'deleveraging': 'sell'},
                    consensus_score=-1.0,  # Special marker for deleveraging
                    risk_checks_passed=True,
                    risk_check_details={'reason': 'margin_deleveraging', 'debt_covered': cash_freed}
                )
                trade.mark_filled(
                    order_result.filled_avg_price,
                    order_result.filled_quantity,
                    order_result.order_id,
                    slippage_dollars=order_result.slippage_dollars,
                    slippage_percentage=order_result.slippage_percentage
                )
                trade.expected_price = current_price
                save_trade(trade)

                logger.info(
                    f"✓ Sold {order_result.filled_quantity} shares of {symbol} @ ${order_result.filled_avg_price:.2f}. "
                    f"Cash freed: ${cash_freed:,.2f}. Total freed: ${cash_freed_total:,.2f} / ${cash_needed:,.2f}"
                )
            else:
                logger.error(
                    f"Failed to sell {symbol} for deleveraging: {order_result.error_message}"
                )
        except Exception as e:
            logger.error(f"Error selling {symbol} for deleveraging: {e}")
            continue

    # Update result
    result['deleveraged'] = True
    result['positions_sold'] = positions_sold
    result['cash_freed'] = cash_freed_total
    result['symbols_sold'] = symbols_sold

    # Send completion alert
    if cash_freed_total >= cash_needed:
        send_alert(
            f"✅ MARGIN DELEVERAGING COMPLETE\n\n"
            f"Margin debt covered: ${margin_debt:,.2f}\n"
            f"Positions sold: {positions_sold}\n"
            f"Symbols: {', '.join(symbols_sold)}\n"
            f"Cash freed: ${cash_freed_total:,.2f}\n\n"
            f"Account is now cash-only (no margin).",
            "INFO"
        )
        logger.info(
            f"Deleveraging complete: Sold {positions_sold} positions, "
            f"freed ${cash_freed_total:,.2f} to cover ${margin_debt:,.2f} margin debt"
        )
    else:
        send_alert(
            f"⚠️ PARTIAL MARGIN DELEVERAGING\n\n"
            f"Margin debt: ${margin_debt:,.2f}\n"
            f"Cash freed: ${cash_freed_total:,.2f}\n"
            f"Remaining debt: ${margin_debt - cash_freed_total:,.2f}\n\n"
            f"Sold all {positions_sold} positions but debt not fully covered.\n"
            f"Manual intervention may be required.",
            "WARNING"
        )
        logger.warning(
            f"Partial deleveraging: Sold all {positions_sold} positions, "
            f"freed ${cash_freed_total:,.2f} but ${margin_debt - cash_freed_total:,.2f} debt remains"
        )

    return result


def check_stop_losses(
    alpaca_client: AlpacaClient,
    risk_manager: RiskManager,
    current_positions: Dict,
    portfolio_value: float,
    historical_data: Dict,
    config: Dict = None,
) -> Dict:
    """
    Check all positions for stop-loss breaches (including trailing stops and breakeven) and execute sells if needed

    Args:
        alpaca_client: AlpacaClient instance
        risk_manager: RiskManager instance
        current_positions: Current positions dict from Alpaca
        portfolio_value: Current portfolio value
        historical_data: Historical market data for all symbols
        config: Trading configuration (for intraday exit rules)

    Returns:
        Dict with stop_losses_triggered, positions_sold, cash_freed
    """
    logger.info("Checking stop-loss levels for all positions (static, trailing, breakeven)...")

    result = {
        'stop_losses_triggered': 0,
        'breakeven_stops_triggered': 0,
        'trailing_stops_triggered': 0,
        'positions_sold': 0,
        'cash_freed': 0.0,
        'details': []
    }

    # Get intraday exit rules config
    intraday_config = config.get('intraday_exit_rules', {}) if config else {}
    intraday_enabled = intraday_config.get('enabled', False)
    trailing_stop_pct = intraday_config.get('trailing_stop_percentage', 0.5)  # 50% of gain by default
    breakeven_threshold_pct = intraday_config.get('breakeven_threshold_percentage', 1.5) / 100.0  # Convert to decimal

    # Get positions from DynamoDB (contains stop_loss_price, peak_price, trailing_stop_active)
    positions_table = dynamodb.Table(POSITIONS_TABLE)

    try:
        db_response = positions_table.scan()
        db_positions = {item['symbol']: item for item in db_response.get('Items', [])}
    except Exception as e:
        logger.error(f"Error reading positions from DynamoDB: {e}")
        return result

    for symbol, position in current_positions.items():
        try:
            # Get stop-loss price from DynamoDB
            db_position = db_positions.get(symbol)
            if not db_position:
                logger.warning(f"No DynamoDB record for position {symbol}, skipping stop-loss check")
                continue

            stop_loss_price = db_position.get('stop_loss_price')
            if not stop_loss_price:
                logger.warning(f"No stop-loss price set for {symbol}, skipping")
                continue

            # Convert Decimal to float
            stop_loss_price = float(stop_loss_price)
            current_price = position['current_price']
            quantity = position['quantity']
            avg_entry_price = float(db_position.get('avg_entry_price', position.get('avg_entry_price', 0)))

            # Get trailing stop settings from DynamoDB
            trailing_stop_active = db_position.get('trailing_stop_active', False)
            peak_price = float(db_position.get('peak_price', avg_entry_price)) if db_position.get('peak_price') else avg_entry_price

            # Calculate profit percentage
            profit_pct = (current_price - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0

            # PDT GROWTH MODE: Skip profit-based stop adjustments (keep original stops for downside protection)
            pdt_growth_mode = config.get('execution', {}).get('pdt_growth_mode', False)

            # BREAKEVEN STOP LOGIC (only if intraday rules enabled and not trailing yet, and NOT in PDT growth mode)
            if intraday_enabled and not trailing_stop_active and not pdt_growth_mode and profit_pct >= breakeven_threshold_pct:
                # Move stop to breakeven (entry price)
                logger.info(
                    f"📈 BREAKEVEN THRESHOLD REACHED: {symbol} @ ${current_price:.2f} "
                    f"(entry: ${avg_entry_price:.2f}, gain: {profit_pct*100:.2f}%) - Moving stop to breakeven"
                )
                stop_loss_price = avg_entry_price
                # Update DynamoDB with new stop
                try:
                    from decimal import Decimal
                    positions_table.update_item(
                        Key={'symbol': symbol},
                        UpdateExpression='SET stop_loss_price = :stop, last_updated = :updated',
                        ExpressionAttributeValues={
                            ':stop': Decimal(str(stop_loss_price)),
                            ':updated': datetime.utcnow().isoformat()
                        }
                    )
                    logger.info(f"✓ Moved stop to breakeven for {symbol}: ${stop_loss_price:.2f}")
                except Exception as e:
                    logger.error(f"Error updating breakeven stop for {symbol}: {e}")

            # TRAILING STOP LOGIC (only if trailing stop active, and NOT in PDT growth mode)
            if intraday_enabled and trailing_stop_active and not pdt_growth_mode:
                # Update peak price if current price is new high
                if current_price > peak_price:
                    peak_price = current_price
                    # Calculate new trailing stop (e.g., 50% of gain from entry)
                    gain_from_entry = peak_price - avg_entry_price
                    new_trailing_stop = avg_entry_price + (gain_from_entry * trailing_stop_pct)

                    # Only move stop up, never down
                    if new_trailing_stop > stop_loss_price:
                        stop_loss_price = new_trailing_stop
                        logger.info(
                            f"⬆️ TRAILING STOP UPDATED: {symbol} peaked @ ${peak_price:.2f}, "
                            f"new stop: ${stop_loss_price:.2f} ({trailing_stop_pct*100:.0f}% of ${gain_from_entry:.2f} gain)"
                        )
                        # Update DynamoDB
                        try:
                            from decimal import Decimal
                            positions_table.update_item(
                                Key={'symbol': symbol},
                                UpdateExpression='SET peak_price = :peak, stop_loss_price = :stop, last_updated = :updated',
                                ExpressionAttributeValues={
                                    ':peak': Decimal(str(peak_price)),
                                    ':stop': Decimal(str(stop_loss_price)),
                                    ':updated': datetime.utcnow().isoformat()
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error updating trailing stop for {symbol}: {e}")

            # Check if stop-loss has been breached
            if current_price <= stop_loss_price:
                # Determine exit reason (trailing, breakeven, or static stop)
                if trailing_stop_active:
                    exit_reason = 'trailing_stop'
                    exit_type_label = "TRAILING STOP"
                    result['trailing_stops_triggered'] += 1
                elif stop_loss_price == avg_entry_price:
                    exit_reason = 'breakeven_stop'
                    exit_type_label = "BREAKEVEN STOP"
                    result['breakeven_stops_triggered'] += 1
                else:
                    exit_reason = 'stop_loss'
                    exit_type_label = "STOP-LOSS"

                logger.warning(
                    f"🛑 {exit_type_label} TRIGGERED: {symbol} @ ${current_price:.2f} "
                    f"(stop: ${stop_loss_price:.2f}, entry: ${avg_entry_price:.2f})"
                )

                result['stop_losses_triggered'] += 1

                # Execute sell order
                order_result = alpaca_client.place_market_order_with_confirmation(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    expected_price=current_price,
                    wait_for_fill=True,
                    max_wait_seconds=30
                )

                if order_result.status == OrderStatus.FILLED:
                    cash_freed = order_result.filled_quantity * order_result.filled_avg_price
                    result['positions_sold'] += 1
                    result['cash_freed'] += cash_freed

                    # Calculate realized P&L
                    realized_pl = (order_result.filled_avg_price - avg_entry_price) * order_result.filled_quantity
                    realized_pl_pct = (realized_pl / (avg_entry_price * order_result.filled_quantity)) * 100

                    logger.info(
                        f"{exit_type_label} sell executed: {order_result.filled_quantity} {symbol} @ "
                        f"${order_result.filled_avg_price:.2f}, realized P&L: ${realized_pl:,.2f} ({realized_pl_pct:+.2f}%)"
                    )

                    # Record trade
                    stop_loss_trade = Trade.create_new(
                        symbol=symbol,
                        side='sell',
                        quantity=quantity,
                        order_type='market',
                        strategy_signals={exit_reason: 'triggered'},
                        consensus_score=-1.0,  # Special marker for stop-loss trades
                        risk_checks_passed=True,
                        risk_check_details={'reason': f'{exit_type_label} triggered'}
                    )
                    stop_loss_trade.mark_filled(
                        order_result.filled_avg_price,
                        order_result.filled_quantity,
                        order_result.order_id,
                        slippage_dollars=order_result.slippage_dollars,
                        slippage_percentage=order_result.slippage_percentage
                    )
                    stop_loss_trade.expected_price = current_price
                    stop_loss_trade.exit_reason = exit_reason
                    stop_loss_trade.calculate_realized_pl(avg_entry_price * order_result.filled_quantity)
                    save_trade(stop_loss_trade)

                    result['details'].append({
                        'symbol': symbol,
                        'entry_price': avg_entry_price,
                        'stop_price': stop_loss_price,
                        'exit_price': order_result.filled_avg_price,
                        'quantity': order_result.filled_quantity,
                        'realized_pl': realized_pl,
                        'realized_pl_pct': realized_pl_pct
                    })

                    # Send alert
                    send_alert(
                        f"🛑 STOP-LOSS EXECUTED - {symbol}\n\n"
                        f"Entry: ${avg_entry_price:.2f}\n"
                        f"Stop: ${stop_loss_price:.2f}\n"
                        f"Exit: ${order_result.filled_avg_price:.2f}\n"
                        f"Quantity: {order_result.filled_quantity}\n"
                        f"Realized P&L: ${realized_pl:,.2f} ({realized_pl_pct:+.2f}%)",
                        "WARNING"
                    )

                else:
                    logger.error(
                        f"Failed to execute stop-loss sell for {symbol}: "
                        f"{order_result.rejection_reason} - {order_result.error_message}"
                    )

        except Exception as e:
            logger.error(f"Error checking stop-loss for {symbol}: {e}", exc_info=True)
            continue

    if result['stop_losses_triggered'] > 0:
        logger.info(
            f"Stop-loss check complete: {result['stop_losses_triggered']} triggered, "
            f"{result['positions_sold']} sold, ${result['cash_freed']:,.2f} freed"
        )
    else:
        logger.info("Stop-loss check complete: No stop-losses triggered")

    return result


def check_take_profit_targets(
    alpaca_client: AlpacaClient,
    risk_manager: RiskManager,
    current_positions: Dict,
    portfolio_value: float,
    config: Dict,
) -> Dict:
    """
    Check positions for take-profit targets and execute partial exits

    Args:
        alpaca_client: AlpacaClient instance
        risk_manager: RiskManager instance
        current_positions: Current positions dict from Alpaca
        portfolio_value: Current portfolio value
        config: Trading configuration

    Returns:
        Dict with take_profits_triggered, positions_sold, cash_freed
    """
    logger.info("Checking take-profit targets for all positions...")

    result = {
        'take_profits_triggered': 0,
        'positions_sold': 0,
        'cash_freed': 0.0,
        'details': []
    }

    # Get intraday exit rules config
    intraday_config = config.get('intraday_exit_rules', {})
    if not intraday_config.get('enabled', False):
        logger.info("Intraday exit rules disabled, skipping take-profit check")
        return result

    # PDT GROWTH MODE: Disable profit-taking to let winners run
    pdt_growth_mode = config.get('execution', {}).get('pdt_growth_mode', False)
    if pdt_growth_mode:
        logger.info("💎 PDT GROWTH MODE: Profit-taking DISABLED - letting winners run to maximize gains until $25K target")
        return result

    # Progressive take-profit ladder settings
    initial_take_profit_pct = intraday_config.get('take_profit_percentage', 6.5) / 100.0  # First level (6.5%)
    take_profit_increment_pct = intraday_config.get('take_profit_increment_percentage', 3.5) / 100.0  # Increment (3.5%)
    partial_exit_pct = intraday_config.get('partial_exit_percentage', 0.33)  # 33% by default
    max_levels = intraday_config.get('max_take_profit_levels', 3)  # Max 3 levels

    # Get positions from DynamoDB (contains entry prices)
    positions_table = dynamodb.Table(POSITIONS_TABLE)

    try:
        db_response = positions_table.scan()
        db_positions = {item['symbol']: item for item in db_response.get('Items', [])}
    except Exception as e:
        logger.error(f"Error reading positions from DynamoDB: {e}")
        return result

    for symbol, position in current_positions.items():
        try:
            # Get entry price from DynamoDB
            db_position = db_positions.get(symbol)
            if not db_position:
                logger.warning(f"No DynamoDB record for position {symbol}, skipping take-profit check")
                continue

            avg_entry_price = float(db_position.get('avg_entry_price', 0))
            if avg_entry_price == 0:
                logger.warning(f"Invalid entry price for {symbol}, skipping")
                continue

            current_price = position['current_price']
            quantity = position['quantity']

            # Calculate profit percentage
            profit_pct = (current_price - avg_entry_price) / avg_entry_price

            # Get current take-profit level from DynamoDB (default to 0 if not set)
            current_level = int(db_position.get('take_profit_level', 0))

            # Check if we've reached max levels
            if current_level >= max_levels:
                continue  # No more take-profit levels for this position

            # Calculate the target for the NEXT level
            # Level 0 → 6.5%, Level 1 → 10.0%, Level 2 → 13.5%
            next_target_pct = initial_take_profit_pct + (current_level * take_profit_increment_pct)

            # Check if take-profit threshold reached for the next level
            if profit_pct >= next_target_pct:
                logger.info(
                    f"💰 TAKE-PROFIT LEVEL {current_level + 1} REACHED: {symbol} @ ${current_price:.2f} "
                    f"(entry: ${avg_entry_price:.2f}, gain: {profit_pct*100:.2f}%, target: {next_target_pct*100:.2f}%)"
                )

                result['take_profits_triggered'] += 1

                # Calculate partial exit quantity (e.g., 50% of position)
                exit_quantity = int(quantity * partial_exit_pct)
                if exit_quantity == 0:
                    logger.warning(f"Partial exit quantity is 0 for {symbol} (quantity: {quantity}), skipping")
                    continue

                # Execute partial sell order
                order_result = alpaca_client.place_market_order_with_confirmation(
                    symbol=symbol,
                    qty=exit_quantity,
                    side='sell',
                    expected_price=current_price,
                    wait_for_fill=True,
                    max_wait_seconds=30
                )

                if order_result.status == OrderStatus.FILLED:
                    cash_freed = order_result.filled_quantity * order_result.filled_avg_price
                    result['positions_sold'] += 1
                    result['cash_freed'] += cash_freed

                    # Calculate realized P&L
                    realized_pl = (order_result.filled_avg_price - avg_entry_price) * order_result.filled_quantity
                    realized_pl_pct = (realized_pl / (avg_entry_price * order_result.filled_quantity)) * 100

                    logger.info(
                        f"Take-profit sell executed: {order_result.filled_quantity}/{quantity} {symbol} @ "
                        f"${order_result.filled_avg_price:.2f}, realized P&L: ${realized_pl:,.2f} ({realized_pl_pct:+.2f}%)"
                    )

                    # Record trade
                    take_profit_trade = Trade.create_new(
                        symbol=symbol,
                        side='sell',
                        quantity=exit_quantity,
                        order_type='market',
                        strategy_signals={'take_profit': 'triggered'},
                        consensus_score=-1.0,  # Special marker for take-profit trades
                        risk_checks_passed=True,
                        risk_check_details={'reason': 'Take-profit target reached'}
                    )
                    take_profit_trade.mark_filled(
                        order_result.filled_avg_price,
                        order_result.filled_quantity,
                        order_result.order_id,
                        slippage_dollars=order_result.slippage_dollars,
                        slippage_percentage=order_result.slippage_percentage
                    )
                    take_profit_trade.expected_price = current_price
                    take_profit_trade.partial_exit = True
                    take_profit_trade.exit_reason = 'take_profit'
                    take_profit_trade.calculate_realized_pl(avg_entry_price * order_result.filled_quantity)
                    save_trade(take_profit_trade)

                    # Update DynamoDB position - reduce quantity, increment level, activate trailing stop
                    remaining_quantity = quantity - order_result.filled_quantity
                    new_level = current_level + 1
                    if remaining_quantity > 0:
                        try:
                            from decimal import Decimal
                            # Update position with reduced quantity, incremented level, and enable trailing stop
                            positions_table.update_item(
                                Key={'symbol': symbol},
                                UpdateExpression='SET quantity = :qty, peak_price = :peak, trailing_stop_active = :active, take_profit_level = :level, last_updated = :updated',
                                ExpressionAttributeValues={
                                    ':qty': Decimal(str(remaining_quantity)),
                                    ':peak': Decimal(str(current_price)),  # Set peak to current price
                                    ':active': True,  # Activate trailing stop for remaining position
                                    ':level': new_level,  # Increment to next take-profit level
                                    ':updated': datetime.utcnow().isoformat()
                                }
                            )
                            next_level_target_pct = initial_take_profit_pct + (new_level * take_profit_increment_pct)
                            logger.info(
                                f"✓ Updated position {symbol}: {remaining_quantity} shares remaining, "
                                f"trailing stop activated @ ${current_price:.2f}, "
                                f"next take-profit level {new_level + 1} @ {next_level_target_pct*100:.1f}%"
                            )
                        except Exception as e:
                            logger.error(f"Error updating position for {symbol} after take-profit: {e}")
                    else:
                        # Fully exited - remove position from DynamoDB
                        try:
                            positions_table.delete_item(Key={'symbol': symbol})
                            logger.info(f"✓ Fully exited {symbol}, removed from DynamoDB")
                        except Exception as e:
                            logger.error(f"Error deleting position {symbol} from DynamoDB: {e}")

                    result['details'].append({
                        'symbol': symbol,
                        'entry_price': avg_entry_price,
                        'exit_price': order_result.filled_avg_price,
                        'quantity_sold': order_result.filled_quantity,
                        'quantity_remaining': remaining_quantity,
                        'realized_pl': realized_pl,
                        'realized_pl_pct': realized_pl_pct
                    })

                    # Send alert
                    send_alert(
                        f"💰 TAKE-PROFIT EXECUTED - {symbol}\n\n"
                        f"Entry: ${avg_entry_price:.2f}\n"
                        f"Exit: ${order_result.filled_avg_price:.2f}\n"
                        f"Gain: {realized_pl_pct:+.2f}%\n"
                        f"Quantity Sold: {order_result.filled_quantity} ({int(partial_exit_pct*100)}% of position)\n"
                        f"Remaining: {remaining_quantity} shares (trailing stop active)\n"
                        f"Realized P&L: ${realized_pl:,.2f}",
                        "INFO"
                    )

                else:
                    logger.error(
                        f"Failed to execute take-profit sell for {symbol}: "
                        f"{order_result.rejection_reason} - {order_result.error_message}"
                    )

        except Exception as e:
            logger.error(f"Error checking take-profit for {symbol}: {e}", exc_info=True)
            continue

    if result['take_profits_triggered'] > 0:
        logger.info(
            f"Take-profit check complete: {result['take_profits_triggered']} triggered, "
            f"{result['positions_sold']} sold, ${result['cash_freed']:,.2f} freed"
        )
    else:
        logger.info("Take-profit check complete: No take-profit targets reached")

    return result


def check_end_of_day_exit(
    alpaca_client: AlpacaClient,
    risk_manager: RiskManager,
    current_positions: Dict,
    portfolio_value: float,
    config: Dict,
    pdt_check_result: Dict = None,
) -> Dict:
    """
    Check if it's time for end-of-day exits with PDT-aware logic

    PDT Growth Mode Strategy:
    - If account >= $25K: Exit all positions (PDT doesn't apply)
    - If account < $25K and NO day trades remaining: Hold ALL positions overnight
    - If account < $25K with day trades available: Only exit significant losses (>threshold)
    - Preserve day trades for winners and small losses to maximize capital growth

    Args:
        alpaca_client: AlpacaClient instance
        risk_manager: RiskManager instance
        current_positions: Current positions dict from Alpaca
        portfolio_value: Current portfolio value
        config: Trading configuration
        pdt_check_result: PDT tracking result with day trades count and limit

    Returns:
        Dict with eod_exit_triggered, positions_closed, cash_freed
    """
    logger.info("Checking if end-of-day exit is needed...")

    result = {
        'eod_exit_triggered': False,
        'positions_closed': 0,
        'cash_freed': 0.0,
        'positions_held_overnight': 0,
        'details': [],
        'pdt_mode': 'disabled'
    }

    # Get intraday exit rules config
    intraday_config = config.get('intraday_exit_rules', {})
    if not intraday_config.get('enabled', False):
        logger.info("Intraday exit rules disabled, skipping EOD exit check")
        return result

    if not intraday_config.get('end_of_day_exit_enabled', False):
        logger.info("End-of-day exit disabled, skipping")
        return result

    # Get EOD exit time (default: 15:45:00 = 3:45 PM ET)
    eod_exit_time_str = intraday_config.get('end_of_day_exit_time', '15:45:00')
    eod_hour, eod_minute, eod_second = map(int, eod_exit_time_str.split(':'))

    # Get current time in ET
    import pytz
    et_tz = pytz.timezone('America/New_York')
    current_time_et = datetime.now(et_tz)

    # Check if current time >= EOD exit time
    eod_exit_time = current_time_et.replace(hour=eod_hour, minute=eod_minute, second=eod_second)

    if current_time_et < eod_exit_time:
        logger.info(f"Current time {current_time_et.strftime('%H:%M:%S')} ET is before EOD exit time {eod_exit_time_str}, skipping")
        return result

    logger.info(f"🔔 END-OF-DAY EXIT TIME REACHED: {current_time_et.strftime('%H:%M:%S')} ET >= {eod_exit_time_str}")

    result['eod_exit_triggered'] = True

    # Check if PDT-aware mode is enabled
    pdt_aware_mode = intraday_config.get('pdt_aware_mode', False)
    pdt_growth_mode = config.get('execution', {}).get('pdt_growth_mode', False)

    # Get positions from DynamoDB (for entry prices)
    positions_table = dynamodb.Table(POSITIONS_TABLE)

    try:
        db_response = positions_table.scan()
        db_positions = {item['symbol']: item for item in db_response.get('Items', [])}
    except Exception as e:
        logger.error(f"Error reading positions from DynamoDB: {e}")
        db_positions = {}

    # PDT-AWARE LOGIC
    if (pdt_aware_mode or pdt_growth_mode) and pdt_check_result:
        account_value = pdt_check_result.get('account_value', portfolio_value)
        day_trades_count = pdt_check_result.get('day_trades_count', 0)
        day_trades_limit = pdt_check_result.get('day_trades_limit', 3)
        remaining_day_trades = day_trades_limit - day_trades_count

        result['pdt_mode'] = 'growth'
        result['account_value'] = account_value
        result['day_trades_remaining'] = remaining_day_trades

        # If account >= $25K, PDT doesn't apply - exit all positions
        if account_value >= 25000:
            logger.info(f"💰 Account value ${account_value:,.2f} >= $25K - PDT EXEMPT! Exiting all positions normally...")
            result['pdt_mode'] = 'exempt'
            # Fall through to normal exit logic below

        # If NO day trades remaining, hold ALL positions overnight
        elif remaining_day_trades == 0:
            logger.warning(f"⚠️  NO DAY TRADES REMAINING (0/{day_trades_limit}) - HOLDING ALL {len(current_positions)} POSITIONS OVERNIGHT!")
            logger.info("Strategy: Preserve capital, avoid PDT violations. Let winners run, hold losers with stops.")
            result['positions_held_overnight'] = len(current_positions)
            result['pdt_mode'] = 'hold_all'

            # Log positions being held
            for symbol, position in current_positions.items():
                entry_price = position.get('avg_entry_price', 0)
                current_price = position['current_price']
                quantity = position['quantity']
                unrealized_pl = position.get('unrealized_pl', 0)
                unrealized_pl_pct = (unrealized_pl / (entry_price * quantity)) * 100 if entry_price > 0 else 0

                logger.info(
                    f"Holding {symbol} overnight: {quantity} shares @ ${current_price:.2f}, "
                    f"P&L: ${unrealized_pl:+,.2f} ({unrealized_pl_pct:+.2f}%)"
                )

            return result

        # Have day trades available - selective exits for significant losses only
        else:
            pdt_loss_exit_threshold = intraday_config.get('pdt_loss_exit_threshold', -2.0)  # Default: -2%
            logger.info(
                f"📊 PDT-AWARE MODE: {remaining_day_trades}/{day_trades_limit} day trades remaining, "
                f"account ${account_value:,.2f}"
            )
            logger.info(f"Strategy: Exit only losses >{abs(pdt_loss_exit_threshold)}%, hold winners/small losses overnight")

            # Sort positions by P&L% to exit worst performers first
            positions_with_pl = []
            for symbol, position in current_positions.items():
                entry_price = position.get('avg_entry_price', 0)
                db_position = db_positions.get(symbol)
                if db_position:
                    entry_price = float(db_position.get('avg_entry_price', entry_price))

                current_price = position['current_price']
                quantity = position['quantity']
                unrealized_pl = (current_price - entry_price) * quantity
                unrealized_pl_pct = (unrealized_pl / (entry_price * quantity)) * 100 if entry_price > 0 else 0

                positions_with_pl.append({
                    'symbol': symbol,
                    'position': position,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'quantity': quantity,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_pl_pct': unrealized_pl_pct
                })

            # Sort by P&L% ascending (worst losses first)
            positions_with_pl.sort(key=lambda p: p['unrealized_pl_pct'])

            # Determine which positions to exit (significant losses only)
            positions_to_exit = []
            for pos_data in positions_with_pl:
                if pos_data['unrealized_pl_pct'] < pdt_loss_exit_threshold:
                    positions_to_exit.append(pos_data)

            # Limit exits to available day trades
            if len(positions_to_exit) > remaining_day_trades:
                logger.warning(
                    f"Want to exit {len(positions_to_exit)} losing positions, "
                    f"but only {remaining_day_trades} day trades available. Exiting worst {remaining_day_trades}."
                )
                positions_to_exit = positions_to_exit[:remaining_day_trades]

            logger.info(f"Selective EOD exits: {len(positions_to_exit)} positions, holding {len(current_positions) - len(positions_to_exit)} overnight")

            # Exit selected positions
            for pos_data in positions_to_exit:
                symbol = pos_data['symbol']
                try:
                    logger.info(
                        f"EOD PDT exit (loss): Selling {pos_data['quantity']} shares of {symbol} @ ${pos_data['current_price']:.2f}, "
                        f"P&L: ${pos_data['unrealized_pl']:+,.2f} ({pos_data['unrealized_pl_pct']:+.2f}%)"
                    )

                    # Execute sell order
                    order_result = alpaca_client.place_market_order_with_confirmation(
                        symbol=symbol,
                        qty=pos_data['quantity'],
                        side='sell',
                        expected_price=pos_data['current_price'],
                        wait_for_fill=True,
                        max_wait_seconds=30
                    )

                    if order_result.status == OrderStatus.FILLED:
                        cash_freed = order_result.filled_quantity * order_result.filled_avg_price
                        result['positions_closed'] += 1
                        result['cash_freed'] += cash_freed

                        # Calculate realized P&L
                        realized_pl = (order_result.filled_avg_price - pos_data['entry_price']) * order_result.filled_quantity
                        realized_pl_pct = (realized_pl / (pos_data['entry_price'] * order_result.filled_quantity)) * 100 if pos_data['entry_price'] > 0 else 0

                        logger.info(
                            f"EOD sell executed: {order_result.filled_quantity} {symbol} @ "
                            f"${order_result.filled_avg_price:.2f}, realized P&L: ${realized_pl:,.2f} ({realized_pl_pct:+.2f}%)"
                        )

                        # Record trade
                        eod_trade = Trade.create_new(
                            symbol=symbol,
                            side='sell',
                            quantity=pos_data['quantity'],
                            order_type='market',
                            strategy_signals={'eod_exit_pdt_loss': 'triggered'},
                            consensus_score=-1.0,
                            risk_checks_passed=True,
                            risk_check_details={'reason': 'EOD PDT exit - significant loss'}
                        )
                        eod_trade.mark_filled(
                            order_result.filled_avg_price,
                            order_result.filled_quantity,
                            order_result.order_id,
                            slippage_dollars=order_result.slippage_dollars,
                            slippage_percentage=order_result.slippage_percentage
                        )
                        eod_trade.expected_price = pos_data['current_price']
                        eod_trade.partial_exit = False
                        eod_trade.exit_reason = 'eod_exit_pdt_loss'
                        eod_trade.calculate_realized_pl(pos_data['entry_price'] * order_result.filled_quantity)
                        save_trade(eod_trade)

                        # Remove position from DynamoDB
                        try:
                            positions_table.delete_item(Key={'symbol': symbol})
                            logger.info(f"✓ Removed {symbol} from DynamoDB (EOD PDT exit)")
                        except Exception as e:
                            logger.error(f"Error deleting position {symbol} from DynamoDB: {e}")

                        result['details'].append({
                            'symbol': symbol,
                            'action': 'exited',
                            'entry_price': pos_data['entry_price'],
                            'exit_price': order_result.filled_avg_price,
                            'quantity': order_result.filled_quantity,
                            'realized_pl': realized_pl,
                            'realized_pl_pct': realized_pl_pct,
                            'reason': 'significant_loss'
                        })
                    else:
                        logger.error(
                            f"Failed to execute EOD sell for {symbol}: "
                            f"{order_result.rejection_reason} - {order_result.error_message}"
                        )

                except Exception as e:
                    logger.error(f"Error executing EOD exit for {symbol}: {e}", exc_info=True)
                    continue

            # Log positions being held overnight
            positions_held = [p for p in positions_with_pl if p not in positions_to_exit]
            result['positions_held_overnight'] = len(positions_held)

            for pos_data in positions_held:
                logger.info(
                    f"Holding {pos_data['symbol']} overnight: {pos_data['quantity']} shares @ ${pos_data['current_price']:.2f}, "
                    f"P&L: ${pos_data['unrealized_pl']:+,.2f} ({pos_data['unrealized_pl_pct']:+.2f}%)"
                )

                result['details'].append({
                    'symbol': pos_data['symbol'],
                    'action': 'held_overnight',
                    'current_price': pos_data['current_price'],
                    'quantity': pos_data['quantity'],
                    'unrealized_pl': pos_data['unrealized_pl'],
                    'unrealized_pl_pct': pos_data['unrealized_pl_pct'],
                    'reason': 'preserve_day_trades'
                })

            return result

    # STANDARD EOD EXIT (not PDT-aware) - Close ALL positions
    logger.info(f"Standard EOD exit: Closing ALL {len(current_positions)} positions before market close...")

    # Close all positions
    for symbol, position in current_positions.items():
        try:
            current_price = position['current_price']
            quantity = position['quantity']
            avg_entry_price = position.get('avg_entry_price', 0)

            # Get entry price from DynamoDB if available
            db_position = db_positions.get(symbol)
            if db_position:
                avg_entry_price = float(db_position.get('avg_entry_price', avg_entry_price))

            logger.info(f"EOD exit: Selling {quantity} shares of {symbol} @ ${current_price:.2f}")

            # Execute sell order
            order_result = alpaca_client.place_market_order_with_confirmation(
                symbol=symbol,
                qty=quantity,
                side='sell',
                expected_price=current_price,
                wait_for_fill=True,
                max_wait_seconds=30
            )

            if order_result.status == OrderStatus.FILLED:
                cash_freed = order_result.filled_quantity * order_result.filled_avg_price
                result['positions_closed'] += 1
                result['cash_freed'] += cash_freed

                # Calculate realized P&L
                realized_pl = (order_result.filled_avg_price - avg_entry_price) * order_result.filled_quantity
                realized_pl_pct = (realized_pl / (avg_entry_price * order_result.filled_quantity)) * 100 if avg_entry_price > 0 else 0

                logger.info(
                    f"EOD sell executed: {order_result.filled_quantity} {symbol} @ "
                    f"${order_result.filled_avg_price:.2f}, realized P&L: ${realized_pl:,.2f} ({realized_pl_pct:+.2f}%)"
                )

                # Record trade
                eod_trade = Trade.create_new(
                    symbol=symbol,
                    side='sell',
                    quantity=quantity,
                    order_type='market',
                    strategy_signals={'eod_exit': 'triggered'},
                    consensus_score=-1.0,  # Special marker for EOD trades
                    risk_checks_passed=True,
                    risk_check_details={'reason': 'End-of-day exit'}
                )
                eod_trade.mark_filled(
                    order_result.filled_avg_price,
                    order_result.filled_quantity,
                    order_result.order_id,
                    slippage_dollars=order_result.slippage_dollars,
                    slippage_percentage=order_result.slippage_percentage
                )
                eod_trade.expected_price = current_price
                eod_trade.partial_exit = False
                eod_trade.exit_reason = 'eod_exit'
                eod_trade.calculate_realized_pl(avg_entry_price * order_result.filled_quantity)
                save_trade(eod_trade)

                # Remove position from DynamoDB
                try:
                    positions_table.delete_item(Key={'symbol': symbol})
                    logger.info(f"✓ Removed {symbol} from DynamoDB (EOD exit)")
                except Exception as e:
                    logger.error(f"Error deleting position {symbol} from DynamoDB: {e}")

                result['details'].append({
                    'symbol': symbol,
                    'entry_price': avg_entry_price,
                    'exit_price': order_result.filled_avg_price,
                    'quantity': order_result.filled_quantity,
                    'realized_pl': realized_pl,
                    'realized_pl_pct': realized_pl_pct
                })

            else:
                logger.error(
                    f"Failed to execute EOD sell for {symbol}: "
                    f"{order_result.rejection_reason} - {order_result.error_message}"
                )

        except Exception as e:
            logger.error(f"Error executing EOD exit for {symbol}: {e}", exc_info=True)
            continue

    if result['positions_closed'] > 0:
        # Calculate aggregate P&L
        total_pl = sum(detail['realized_pl'] for detail in result['details'])

        logger.info(
            f"End-of-day exit complete: {result['positions_closed']} positions closed, "
            f"${result['cash_freed']:,.2f} freed, total P&L: ${total_pl:,.2f}"
        )

        # Send summary alert
        winners = [d for d in result['details'] if d['realized_pl'] > 0]
        losers = [d for d in result['details'] if d['realized_pl'] < 0]

        send_alert(
            f"🔔 END-OF-DAY EXIT COMPLETE\n\n"
            f"Positions Closed: {result['positions_closed']}\n"
            f"Winners: {len(winners)}\n"
            f"Losers: {len(losers)}\n"
            f"Total P&L: ${total_pl:,.2f}\n"
            f"Cash Freed: ${result['cash_freed']:,.2f}\n\n"
            f"All positions closed before market close.",
            "INFO"
        )
    else:
        logger.info("End-of-day exit complete: No positions to close")

    return result


def save_trade(trade: Trade):
    """Save trade to DynamoDB"""
    table = dynamodb.Table(TRADES_TABLE)
    table.put_item(Item=trade.to_dynamodb_item())


def get_closed_trades(days_back: int = 30) -> List[Dict]:
    """
    Get closed trades from the past N days for wash sale tracking

    Returns list of dicts with: symbol, side, realized_pl, timestamp
    """
    table = dynamodb.Table(TRADES_TABLE)

    try:
        # Calculate cutoff date
        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        # Scan for filled sell trades with realized_pl in the past N days
        # Note: This is a scan operation which can be expensive for large tables
        # For production, consider adding a GSI on timestamp for efficient queries
        response = table.scan(
            FilterExpression='#status = :filled AND #side = :sell AND #timestamp > :cutoff',
            ExpressionAttributeNames={
                '#status': 'status',
                '#side': 'side',
                '#timestamp': 'timestamp'
            },
            ExpressionAttributeValues={
                ':filled': 'filled',
                ':sell': 'sell',
                ':cutoff': cutoff_date
            }
        )

        trades = []
        for item in response.get('Items', []):
            # Only include trades with realized_pl (completed sells)
            if 'realized_pl' in item and item['realized_pl'] is not None:
                trades.append({
                    'symbol': item.get('symbol'),
                    'side': item.get('side'),
                    'realized_pl': float(item.get('realized_pl')),
                    'timestamp': item.get('timestamp')
                })

        logger.info(f"Retrieved {len(trades)} closed trades from past {days_back} days")
        return trades

    except Exception as e:
        logger.error(f"Error fetching closed trades: {e}")
        return []


def get_latest_risk_metrics() -> Dict:
    """Get latest risk metrics from DynamoDB"""
    table = dynamodb.Table(RISK_METRICS_TABLE)

    try:
        response = table.query(
            KeyConditionExpression='metric_type = :mt',
            ExpressionAttributeValues={':mt': 'daily'},
            ScanIndexForward=False,
            Limit=1
        )

        if response['Items']:
            return response['Items'][0]

    except Exception as e:
        logger.warning(f"Error getting risk metrics: {e}")

    return {}


def update_risk_metrics(alpaca_client, portfolio_value, current_positions):
    """Update risk metrics in DynamoDB"""
    # This is a simplified version
    # In production, you'd calculate all metrics properly

    table = dynamodb.Table(RISK_METRICS_TABLE)

    metrics = RiskMetrics.create_snapshot(
        metric_type='daily',
        portfolio_value=portfolio_value,
        cash_balance=0.0,  # Would get from account
        total_exposure=sum(p['market_value'] for p in current_positions.values()),
        daily_pl=0.0,  # Would calculate
        weekly_pl=0.0,  # Would calculate
        max_drawdown=0.0,  # Would calculate
        sector_concentrations={},
        position_count=len(current_positions),
        active_strategies=[],
        win_rate=0.5,  # Would calculate
        consecutive_losses=0,  # Would calculate
        vix_level=15.0,  # Would fetch
        circuit_breaker_active=False,
        risk_violations=[]
    )

    table.put_item(Item=metrics.to_dynamodb_item())


def activate_circuit_breaker(reason: str):
    """Activate circuit breaker"""
    table = dynamodb.Table(SYSTEM_STATE_TABLE)

    state = SystemState.create(
        state_key='circuit_breaker',
        value=True,
        updated_by='system',
        reason=reason
    )

    table.put_item(Item=state.to_dynamodb_item())

    send_alert(f"CIRCUIT BREAKER ACTIVATED: {reason}", "CRITICAL")
    logger.critical(f"Circuit breaker activated: {reason}")


def send_alert(message: str, level: str = "INFO"):
    """Send SNS alert"""
    try:
        sns.publish(
            TopicArn=ALERT_TOPIC_ARN,
            Subject=f"Trading Bot Alert [{level}]",
            Message=message
        )
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
