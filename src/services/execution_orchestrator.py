"""
Execution Orchestrator

Coordinates all trading services to execute the complete trading cycle.
This is the main orchestration layer that replaces the monolithic handler logic.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import json
import os
import logging
import uuid
import boto3
import pandas as pd
from alpaca.data.timeframe import TimeFrame

from alpaca_client.client import AlpacaClient
from market_scanner.scanner import MarketScanner
from strategies.strategy_manager import StrategyManager
from risk_management.risk_manager import RiskManager
from utils.position_reconciliation import PositionReconciler
from position_management.position_evaluator import PositionEvaluator
from utils.pdt_tracker import PDTTracker
from utils.wash_sale_tracker import WashSaleTracker
from models.risk_metrics import RiskMetrics
from tradingbot_core.models.utils import convert_floats_to_decimal
from utils.logger import setup_logger
from utils.type_conversion import safe_float, safe_int

# Shared services and models from tradingbot_core
from tradingbot_core.services import (
    BalanceService,
    BalanceTracker,
    MarketDataService,
    SystemStateService,
    TaxService,
)
from tradingbot_core import Trade
from tradingbot_core.utils import TechnicalIndicators

# Bot-specific services (local)
from services.pdt_service import PDTService
from services.order_service import OrderService
from utils.config_loader import load_config_from_s3

logger = setup_logger(__name__)

# AWS resources
dynamodb = boto3.resource('dynamodb')
ssm = boto3.client('ssm')


class ExecutionContext:
    """Context for a single execution cycle"""
    def __init__(self, event: Dict, context: Any):
        self.event = event
        self.lambda_context = context
        self.execution_id = context.aws_request_id if context else 'local'
        self.start_time = datetime.utcnow()
        self.config = {}
        self.use_margin = False


class ExecutionOrchestrator:
    """
    Main orchestrator that coordinates all trading services.

    This class replaces the monolithic handler by delegating to specialized services.
    """

    def __init__(self):
        """Initialize the orchestrator (services initialized per execution)"""
        # Environment variables with validation
        self.environment = self._validate_env_var(os.environ.get('ENVIRONMENT', 'dev'), 'ENVIRONMENT')
        self.profile = self._validate_env_var(os.environ.get('PROFILE', 'aggressive'), 'PROFILE')
        self.stage = self._validate_env_var(os.environ.get('STAGE', 'blue'), 'STAGE')

        # DynamoDB table names
        self.positions_table_name = os.environ['POSITIONS_TABLE']
        self.trades_table_name = os.environ['TRADES_TABLE']
        self.risk_metrics_table_name = os.environ['RISK_METRICS_TABLE']
        self.system_state_table_name = os.environ['SYSTEM_STATE_TABLE']
        self.day_trades_table_name = os.environ.get('DAY_TRADES_TABLE', '')
        self.realized_losses_table_name = os.environ.get('REALIZED_LOSSES_TABLE', '')
        self.universe_cache_table_name = os.environ.get('UNIVERSE_CACHE_TABLE', '')
        self.tax_obligations_table_name = os.environ.get('TAX_OBLIGATIONS_TABLE', '')

        # S3 bucket name (includes account ID suffix from CDK)
        self.data_bucket_name = os.environ.get('DATA_BUCKET', '')

        logger.info(
            f"ExecutionOrchestrator initialized: {self.environment}/{self.profile}/{self.stage}"
        )

    def execute(self, event: Dict, context: Any) -> Dict:
        """
        Main execution entry point

        Args:
            event: Lambda event
            context: Lambda context

        Returns:
            Lambda response dict
        """
        exec_context = ExecutionContext(event, context)
        logger.info(f"Starting execution: {exec_context.execution_id}")

        try:
            # Initialize services
            services = self._initialize_services(exec_context)

            # Pre-execution checks
            can_trade, reason = self._pre_execution_checks(services)
            if not can_trade:
                return self._format_response(False, reason, {})

            # Execute main trading cycle
            results = self._execute_trading_cycle(services, exec_context)

            # Format and return success response
            return self._format_response(True, "Execution complete", results)

        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)
            return self._format_error_response(str(e))

    def _initialize_services(self, exec_context: ExecutionContext) -> Dict:
        """
        Initialize all services and components

        Args:
            exec_context: Execution context

        Returns:
            Dict of initialized services and components
        """
        logger.info("Initializing services...")

        # Get Alpaca client
        alpaca_client = self._get_alpaca_client()

        # Load configuration
        config = self._load_config()
        exec_context.config = config

        # Get use_margin setting from Parameter Store
        use_margin = self._get_use_margin_setting()
        exec_context.use_margin = use_margin

        # Initialize core services
        market_data_service = MarketDataService(alpaca_client, config)
        system_state_service = SystemStateService(dynamodb, self.system_state_table_name)
        balance_service = BalanceService(use_margin, config)
        pdt_service = PDTService(config, dynamodb)

        # Initialize tax service
        tax_config = config.get('tax_tracking', {})
        tax_service = None
        if self.tax_obligations_table_name and tax_config.get('enabled', True):
            tax_service = TaxService(
                dynamodb_client=dynamodb,
                tax_obligations_table_name=self.tax_obligations_table_name,
                positions_table_name=self.positions_table_name,
                trades_table_name=self.trades_table_name,
                short_term_tax_rate=tax_config.get('short_term_rate', 0.24),
                long_term_tax_rate=tax_config.get('long_term_rate', 0.15)
            )

        # Initialize order service (with tax service integration)
        order_service = OrderService(
            alpaca_client,
            dynamodb,
            self.trades_table_name,
            self.positions_table_name,
            tax_service
        )

        # Initialize existing managers
        # Pass universe cache table to MarketScanner if available
        universe_cache_table = None
        if self.universe_cache_table_name:
            universe_cache_table = dynamodb.Table(self.universe_cache_table_name)

        market_scanner = MarketScanner(
            alpaca_client,
            config=config.get('market_scanner', {}),
            dynamodb_table=universe_cache_table
        )
        strategy_manager = StrategyManager(
            consensus_threshold=config.get('trading', {}).get('consensus_threshold', 0.20),
            config=config.get('strategies', {})
        )
        risk_manager = RiskManager(
            config=config.get('risk_management', {}),
            use_margin=use_margin
        )
        position_evaluator = PositionEvaluator(
            config=config.get('position_management', {})
        )

        # Initialize wash sale tracker if enabled
        wash_sale_tracker = None
        wash_sale_enabled = config.get('tax_compliance', {}).get('wash_sale_tracking_enabled', False)
        wash_sale_period = config.get('tax_compliance', {}).get('wash_sale_period_days', 30)
        wash_sale_tracker = WashSaleTracker(
            enabled=wash_sale_enabled,
            wash_sale_period_days=wash_sale_period
        )

        logger.info("All services initialized successfully")

        return {
            'alpaca_client': alpaca_client,
            'config': config,
            'use_margin': use_margin,
            'dynamodb': dynamodb,
            'dynamodb_client': dynamodb,  # Alias for consistency
            'market_data_service': market_data_service,
            'system_state_service': system_state_service,
            'balance_service': balance_service,
            'pdt_service': pdt_service,
            'order_service': order_service,
            'market_scanner': market_scanner,
            'strategy_manager': strategy_manager,
            'risk_manager': risk_manager,
            'position_evaluator': position_evaluator,
            'wash_sale_tracker': wash_sale_tracker
        }

    def _pre_execution_checks(self, services: Dict) -> tuple[bool, Optional[str]]:
        """
        Perform pre-execution checks

        Args:
            services: Dict of services

        Returns:
            Tuple of (can_proceed, reason_if_not)
        """
        alpaca_client = services['alpaca_client']
        system_state_service = services['system_state_service']

        # Check if market is open
        if not alpaca_client.is_market_open():
            logger.info("Market is closed, skipping execution")
            return False, "Market is closed"

        # Check system state
        can_trade, reason = system_state_service.can_trade()
        if not can_trade:
            logger.warning(f"Cannot trade: {reason}")
            return False, reason

        return True, None

    def _execute_trading_cycle(self, services: Dict, exec_context: ExecutionContext) -> Dict:
        """
        Execute the main trading cycle

        Args:
            services: Dict of services
            exec_context: Execution context

        Returns:
            Dict with execution results
        """
        alpaca_client = services['alpaca_client']
        config = services['config']
        market_scanner = services['market_scanner']
        strategy_manager = services['strategy_manager']
        risk_manager = services['risk_manager']
        dynamodb_client = services['dynamodb']

        # Get account info
        account = alpaca_client.get_account()
        portfolio_value = safe_float(account.equity, default=0.0, field_name="account.equity")
        buying_power = safe_float(account.buying_power, default=0.0, field_name="account.buying_power")
        cash_balance = safe_float(account.cash, default=0.0, field_name="account.cash")

        logger.info(
            f"Portfolio value: ${portfolio_value:,.2f}, "
            f"Buying power: ${buying_power:,.2f}, Cash: ${cash_balance:,.2f}"
        )

        # Get current positions
        current_positions = self._get_current_positions(alpaca_client)

        # Reconcile positions (ensure DynamoDB matches Alpaca)
        self._reconcile_positions(alpaca_client)

        # Initialize balance tracker
        balance_tracker = services['balance_service'].create_balance_tracker(
            cash_balance, buying_power
        )

        # Get PDT status
        pdt_status = None
        if config.get('pdt_tracking', {}).get('enabled', True):
            recent_trades = self._get_recent_trades_for_pdt()
            pdt_status = services['pdt_service'].get_pdt_status(
                portfolio_value, recent_trades, current_positions
            )

        # Check for deleveraging needs
        deleverage_result = self._check_deleveraging(
            services, cash_balance, current_positions, portfolio_value
        )
        if deleverage_result['deleveraged']:
            cash_balance += deleverage_result['cash_freed']
            balance_tracker.cash_balance = cash_balance
            # Refresh positions after deleveraging
            current_positions = self._get_current_positions(alpaca_client)

        # Main trading logic
        summary = {
            'buy_signals': 0,
            'sell_signals': 0,
            'trades_executed': 0,
            'trades_rejected': 0,
            'errors': 0,
            'deleveraged': deleverage_result['deleveraged'],
            'deleverage_positions_sold': deleverage_result['positions_sold']
        }

        # Get trading universe
        trading_universe = self._get_trading_universe(
            services['market_scanner'],
            services['config'],
            current_positions
        )

        logger.info(f"Trading universe: {len(trading_universe)} symbols")

        # Get PDT growth mode setting
        pdt_growth_mode = config.get('execution', {}).get('pdt_growth_mode', False)

        # Get use_margin setting from execution context
        use_margin = exec_context.use_margin

        # PHASE 1: PRE-TRADING CHECKS (CRITICAL - must run first)
        logger.info("=" * 60)
        logger.info("PHASE 1: PRE-TRADING CHECKS")
        logger.info("=" * 60)

        # 1. EOD Exit Check (highest priority)
        eod_result = self._check_end_of_day_exit(
            services, current_positions, portfolio_value,
            pdt_status, balance_tracker, use_margin
        )

        if eod_result['eod_exit_triggered']:
            summary['eod_exits'] = eod_result['positions_exited']
            summary['positions_held_overnight'] = eod_result['positions_held_overnight']

            # Refresh positions and balances after EOD exits
            current_positions = self._refresh_positions(alpaca_client)
            cash_balance, buying_power = self._update_balances_after_sells(alpaca_client, use_margin)
            balance_tracker = BalanceTracker(cash_balance, buying_power, use_margin)

            logger.info(f"EOD exits complete: {eod_result['positions_exited']} exited, {eod_result['positions_held_overnight']} held overnight")

            # If EOD exit triggered, skip normal trading
            return summary

        # 2. Stop-Loss Check
        stop_result = self._check_stop_losses(
            services, current_positions, portfolio_value,
            balance_tracker, use_margin, pdt_growth_mode
        )

        if stop_result['stop_losses_triggered'] > 0:
            summary['stop_losses_triggered'] = stop_result['stop_losses_triggered']

            # Refresh positions and balances
            current_positions = self._refresh_positions(alpaca_client)
            cash_balance, buying_power = self._update_balances_after_sells(alpaca_client, use_margin)
            balance_tracker = BalanceTracker(cash_balance, buying_power, use_margin)

        # 3. Take-Profit Check
        tp_result = self._check_take_profit_targets(
            services, current_positions, portfolio_value,
            balance_tracker, use_margin, pdt_growth_mode
        )

        if tp_result['take_profits_triggered'] > 0:
            summary['take_profits_triggered'] = tp_result['take_profits_triggered']

            # Refresh positions and balances
            current_positions = self._refresh_positions(alpaca_client)
            cash_balance, buying_power = self._update_balances_after_sells(alpaca_client, use_margin)
            balance_tracker = BalanceTracker(cash_balance, buying_power, use_margin)

        # PHASE 2: MARKET DATA PREPARATION
        logger.info("=" * 60)
        logger.info("PHASE 2: MARKET DATA PREPARATION")
        logger.info("=" * 60)

        # 1. Fetch historical data (300 days daily)
        market_data_service = services['market_data_service']
        historical_data = self._fetch_historical_data(
            market_data_service, trading_universe, current_positions
        )

        if not historical_data:
            logger.warning("No historical data available, cannot trade")
            return summary

        # 2. Fetch intraday data (5-min bars, cached)
        intraday_data = self._fetch_intraday_data(
            market_data_service, trading_universe, current_positions, config
        )

        # 3. Apply pre-filter
        if config.get('market_scanner', {}).get('pre_filter_enabled', True):
            trading_universe = self._apply_pre_filter(
                market_scanner, trading_universe, historical_data, intraday_data
            )

        # 4. Always include held positions
        trading_universe = self._ensure_positions_in_universe(trading_universe, current_positions)

        logger.info(f"Final trading universe: {len(trading_universe)} symbols")

        # PHASE 3: MAIN TRADING LOOP (Two-Pass Approach)
        logger.info("=" * 60)
        logger.info("PHASE 3: MAIN TRADING LOOP")
        logger.info("=" * 60)

        strategy_manager = services['strategy_manager']
        consensus_threshold = config.get('trading', {}).get('consensus_threshold', 0.15)

        # ----------------------------------------------------------------
        # Pass 1: Collect all signals (BUY and SELL)
        # ----------------------------------------------------------------
        logger.info("--- Pass 1: Collecting signals ---")
        buy_signals = []  # List of {symbol, consensus_score, strategy_signals}
        sell_signals = []  # List of {symbol, consensus_score, strategy_signals}

        for symbol in trading_universe:
            # Get data
            data = historical_data.get(symbol)
            if data is None or data.empty:
                continue

            symbol_intraday = intraday_data.get(symbol)
            spy_intraday = intraday_data.get('SPY')
            spy_data = historical_data.get('SPY')

            # Generate consensus signal
            try:
                action, consensus_score, strategy_signals = strategy_manager.generate_consensus_signal(
                    symbol=symbol,
                    data=data,
                    spy_data=spy_data,
                    intraday_data=symbol_intraday,
                    spy_intraday_data=spy_intraday
                )
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue

            # Log analysis for every symbol
            strategy_votes = ', '.join([f"{name}: {sig['action']}" for name, sig in strategy_signals.items()])
            logger.info(f"📊 Analyzing {symbol}: action={action.upper()}, consensus={consensus_score:.2f} (threshold={consensus_threshold:.2f})")
            logger.info(f"   Strategy votes: {strategy_votes}")

            # Collect BUY signals
            if action == 'buy' and abs(consensus_score) >= consensus_threshold:
                logger.info(f"   ✅ Decision: BUY signal generated (meets {consensus_threshold:.2f} threshold)")
                buy_signals.append({
                    'symbol': symbol,
                    'consensus_score': consensus_score,
                    'strategy_signals': strategy_signals
                })

            # Collect SELL signals
            elif action == 'sell' and abs(consensus_score) >= consensus_threshold:
                if symbol in current_positions:
                    logger.info(f"   ✅ Decision: SELL signal generated (meets {consensus_threshold:.2f} threshold)")
                    sell_signals.append({
                        'symbol': symbol,
                        'consensus_score': consensus_score,
                        'strategy_signals': strategy_signals
                    })
                else:
                    logger.info(f"   ⏭️  Decision: SKIP - SELL signal but no position held")

            elif action == 'hold':
                logger.info(f"   ⏸️  Decision: HOLD - No clear signal from strategies")
            else:
                logger.info(f"   ❌ Decision: SKIP - Consensus {consensus_score:.2f} below {consensus_threshold:.2f} threshold")

        # Sort buy signals by consensus score (strongest first)
        buy_signals.sort(key=lambda x: x['consensus_score'], reverse=True)

        logger.info(f"--- Signals collected: {len(buy_signals)} BUY, {len(sell_signals)} SELL ---")
        summary['buy_signals'] = len(buy_signals)
        summary['sell_signals'] = len(sell_signals)

        # ----------------------------------------------------------------
        # Pass 2: Execute SELL signals first (to free up cash)
        # ----------------------------------------------------------------
        logger.info("--- Pass 2: Executing SELL signals ---")
        for signal in sell_signals:
            symbol = signal['symbol']
            logger.info(f"🔴 SELL signal: {symbol} (consensus: {signal['consensus_score']:.2f})")

            sell_result = self._execute_sell(
                services=services,
                symbol=symbol,
                position=current_positions[symbol],
                historical_data=historical_data,
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                consensus_score=signal['consensus_score'],
                strategy_signals=signal['strategy_signals'],
                balance_tracker=balance_tracker,
                use_margin=use_margin
            )

            if sell_result['executed']:
                summary['trades_executed'] += 1
                logger.info(f"  ✅ Sell executed: {sell_result['quantity']} @ ${sell_result['price']:.2f} (P/L: ${sell_result['pl']:+,.2f})")

                # Update balance tracker
                proceeds = sell_result['quantity'] * sell_result['price']
                balance_tracker.update_after_sell(proceeds, symbol, sell_result['quantity'])

                # Remove from current positions
                if symbol in current_positions:
                    del current_positions[symbol]
            else:
                summary['trades_rejected'] += 1
                logger.warning(f"  ❌ Sell rejected: {sell_result.get('reason', 'Unknown')}")

        # ----------------------------------------------------------------
        # Pass 3: Execute BUY signals (strongest first, with full context)
        # ----------------------------------------------------------------
        logger.info("--- Pass 3: Executing BUY signals ---")
        total_buy_signals = len(buy_signals)

        for signal in buy_signals:
            symbol = signal['symbol']
            logger.info(f"🟢 BUY signal: {symbol} (consensus: {signal['consensus_score']:.2f})")

            buy_result = self._execute_buy(
                services=services,
                symbol=symbol,
                historical_data=historical_data,
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                consensus_score=signal['consensus_score'],
                strategy_signals=signal['strategy_signals'],
                balance_tracker=balance_tracker,
                use_margin=use_margin,
                pdt_status=pdt_status,
                total_buy_signals=total_buy_signals
            )

            if buy_result['executed']:
                summary['trades_executed'] += 1
                logger.info(f"  ✅ Buy executed: {buy_result['quantity']} @ ${buy_result['price']:.2f}")

                # Update balance tracker
                cost = buy_result['quantity'] * buy_result['price']
                balance_tracker.update_after_buy(cost, symbol, buy_result['quantity'])

                # Add to current positions
                current_positions[symbol] = {
                    'symbol': symbol,
                    'quantity': buy_result['quantity'],
                    'avg_entry_price': buy_result['price'],
                    'current_price': buy_result['price'],
                    'stop_loss_price': buy_result.get('stop_loss_price', 0),
                    'unrealized_pl': 0,
                    'market_value': buy_result['quantity'] * buy_result['price'],
                    'side': 'long'
                }
            else:
                summary['trades_rejected'] += 1
                logger.warning(f"  ❌ Buy rejected: {buy_result.get('reason', 'Unknown')}")

        # PHASE 4: POST-TRADING
        logger.info("=" * 60)
        logger.info("PHASE 4: POST-TRADING")
        logger.info("=" * 60)

        # 1. Update position tracking (peak/trough price and P&L)
        self._update_position_tracking(
            dynamodb_client, self.positions_table_name,
            current_positions
        )

        # 2. Update risk metrics
        self._update_risk_metrics(
            dynamodb_client, self.risk_metrics_table_name,
            portfolio_value, current_positions
        )

        # 3. Check circuit breaker
        risk_metrics = self._get_latest_risk_metrics(dynamodb_client, self.risk_metrics_table_name)
        if risk_metrics and risk_manager.should_activate_circuit_breaker(risk_metrics):
            logger.critical("🚨 CIRCUIT BREAKER: Risk limits breached - activating circuit breaker")
            system_state_service = services['system_state_service']
            system_state_service.activate_circuit_breaker("Risk limits breached")
            summary['circuit_breaker_activated'] = True

        # Log final summary
        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Buy signals: {summary['buy_signals']}")
        logger.info(f"Sell signals: {summary['sell_signals']}")
        logger.info(f"Trades executed: {summary['trades_executed']}")
        logger.info(f"Trades rejected: {summary['trades_rejected']}")
        logger.info(f"Stop losses: {summary.get('stop_losses_triggered', 0)}")
        logger.info(f"Take profits: {summary.get('take_profits_triggered', 0)}")
        logger.info(f"EOD exits: {summary.get('eod_exits', 0)}")
        logger.info("=" * 60)

        return summary

    def _get_trading_universe(
        self,
        market_scanner: MarketScanner,
        config: Dict,
        current_positions: Dict
    ) -> List[str]:
        """
        Get the trading universe (symbols to analyze)

        Args:
            market_scanner: MarketScanner instance
            config: Configuration dict
            current_positions: Current positions

        Returns:
            List of symbols to analyze
        """
        # Get blacklist
        blacklist = self._get_blacklist()

        # Get trading universe parameters from config
        scanner_config = config.get('market_scanner', {})
        min_price = float(scanner_config.get('min_stock_price', 5.0))
        max_price = float(scanner_config.get('max_stock_price', 1000.0))
        min_volume = int(scanner_config.get('min_daily_volume', 500000))
        max_symbols = int(scanner_config.get('max_universe_size', 200))

        # Discover trading universe
        discovered_universe = market_scanner.get_tradeable_universe(
            min_price=min_price,
            max_price=max_price,
            min_volume=min_volume,
            max_symbols=max_symbols,
            exclude_symbols=blacklist
        )

        # Always include held positions
        position_symbols = list(current_positions.keys())

        # Combine and deduplicate
        all_symbols = list(set(discovered_universe + position_symbols))

        logger.info(
            f"Trading universe: {len(discovered_universe)} discovered + "
            f"{len(position_symbols)} held = {len(all_symbols)} total"
        )

        return all_symbols

    def _check_deleveraging(
        self,
        services: Dict,
        cash_balance: float,
        current_positions: Dict,
        portfolio_value: float
    ) -> Dict:
        """
        Check if deleveraging is needed and execute if necessary

        Args:
            services: Dict of services
            cash_balance: Current cash balance
            current_positions: Current positions
            portfolio_value: Portfolio value

        Returns:
            Dict with deleveraging results
        """
        balance_service = services['balance_service']

        has_debt, debt_amount = balance_service.detect_margin_debt(cash_balance)

        if not has_debt:
            return {
                'deleveraged': False,
                'positions_sold': 0,
                'cash_freed': 0.0,
                'symbols_sold': []
            }

        logger.warning(f"Margin deleveraging required: ${debt_amount:,.2f}")

        # For now, return placeholder
        # Full implementation would score positions and sell weakest
        return {
            'deleveraged': False,
            'positions_sold': 0,
            'cash_freed': 0.0,
            'symbols_sold': []
        }

    def _reconcile_positions(self, alpaca_client: AlpacaClient) -> None:
        """
        Reconcile positions between DynamoDB and Alpaca

        Args:
            alpaca_client: AlpacaClient instance
        """
        positions_table = dynamodb.Table(self.positions_table_name)
        reconciler = PositionReconciler(alpaca_client, positions_table)
        discrepancies = reconciler.reconcile()

        if discrepancies:
            logger.info(f"Position reconciliation synced {len(discrepancies)} discrepancy(ies)")
            for disc in discrepancies:
                logger.info(f"  - {disc['type']}: {disc['symbol']} - {disc['details']}")

    def _get_current_positions(self, alpaca_client: AlpacaClient) -> Dict:
        """
        Get current positions from Alpaca

        Args:
            alpaca_client: AlpacaClient instance

        Returns:
            Dict mapping symbol to position data
        """
        try:
            positions = alpaca_client.get_positions()
            position_dict = {}

            for position in positions:
                position_dict[position.symbol] = {
                    'symbol': position.symbol,
                    'quantity': safe_float(position.qty, default=0.0, field_name=f"{position.symbol}.qty"),
                    'avg_entry_price': safe_float(position.avg_entry_price, default=0.0, field_name=f"{position.symbol}.avg_entry_price"),
                    'current_price': safe_float(position.current_price, default=0.0, field_name=f"{position.symbol}.current_price"),
                    'market_value': safe_float(position.market_value, default=0.0, field_name=f"{position.symbol}.market_value"),
                    'unrealized_pl': safe_float(position.unrealized_pl, default=0.0, field_name=f"{position.symbol}.unrealized_pl"),
                    'unrealized_pl_pct': safe_float(position.unrealized_plpc, default=0.0, field_name=f"{position.symbol}.unrealized_plpc"),
                    'side': position.side
                }

            logger.info(f"Retrieved {len(position_dict)} current positions")
            return position_dict

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def _get_recent_trades_for_pdt(self) -> List[Dict]:
        """
        Get recent trades for PDT tracking (last 7 days)

        Returns:
            List of recent trades
        """
        try:
            trades_table = dynamodb.Table(self.trades_table_name)
            lookback_date = (datetime.utcnow() - timedelta(days=7)).isoformat()

            response = trades_table.scan(
                FilterExpression='#ts >= :lookback',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':lookback': lookback_date}
            )

            return response.get('Items', [])

        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []

    def _get_blacklist(self) -> List[str]:
        """
        Get symbol blacklist from S3

        Returns:
            List of blacklisted symbols
        """
        try:
            s3 = boto3.client('s3')
            # Use bucket name from environment variable (includes account ID suffix)
            bucket = self.data_bucket_name
            key = "blacklist.json"

            if not bucket:
                logger.warning("DATA_BUCKET environment variable not set, skipping blacklist load")
                return []

            obj = s3.get_object(Bucket=bucket, Key=key)
            blacklist = json.loads(obj['Body'].read().decode('utf-8'))

            logger.info(f"Loaded blacklist: {len(blacklist)} symbols")
            return blacklist

        except Exception as e:
            # Silently handle "file doesn't exist" - this is expected when no blacklist is configured
            if 'NoSuchKey' in str(e):
                return []
            # Only warn on actual errors (permissions, etc.)
            logger.warning(f"Could not load blacklist: {e}")
            return []

    def _get_alpaca_client(self) -> AlpacaClient:
        """
        Get Alpaca client with credentials from Secrets Manager

        Returns:
            AlpacaClient instance
        """
        secret_name = os.environ.get(
            'ALPACA_SECRET_NAME',
            f'trading-bot/alpaca-credentials-{self.environment}-{self.profile}-{self.stage}'
        )

        try:
            import boto3
            secrets_manager = boto3.client('secretsmanager', region_name='us-east-2')
            secret = secrets_manager.get_secret_value(SecretId=secret_name)
            secret_data = json.loads(secret['SecretString'])

            api_key = secret_data['api_key']
            secret_key = secret_data['secret_key']
            paper = secret_data.get('paper', True)

            return AlpacaClient(api_key, secret_key, paper)

        except Exception as e:
            logger.error(f"Error getting Alpaca credentials from {secret_name}: {e}")
            raise

    def _load_config(self) -> Dict:
        """
        Load trading configuration from S3

        Returns:
            Configuration dict
        """
        try:
            # Load config from S3 using utility function
            config = load_config_from_s3(use_cache=True)
            logger.info(f"Loaded trading config from S3: {len(config)} sections")
            return config
        except Exception as e:
            logger.error(f"Failed to load trading config from S3: {e}")
            # Try fallback to legacy TRADING_CONFIG env var (for backward compatibility)
            try:
                config_json = os.environ.get('TRADING_CONFIG')
                if config_json:
                    logger.warning("Falling back to TRADING_CONFIG environment variable")
                    config = json.loads(config_json)
                    logger.info(f"Loaded trading config from env var: {len(config)} sections")
                    return config
            except Exception as fallback_error:
                logger.error(f"Fallback to env var also failed: {fallback_error}")

            return {}

    def _get_use_margin_setting(self) -> bool:
        """
        Get use_margin setting from AWS Systems Manager Parameter Store

        Returns:
            True if margin enabled, False otherwise
        """
        try:
            parameter_name = f'/trading-bot/{self.environment}/{self.profile}/{self.stage}/use_margin'
            response = ssm.get_parameter(Name=parameter_name)
            value = response['Parameter']['Value'].lower()
            use_margin = value in ['true', '1', 'yes']

            logger.info(f"use_margin setting: {use_margin}")
            return use_margin

        except ssm.exceptions.ParameterNotFound:
            logger.warning(f"Parameter {parameter_name} not found, defaulting to False")
            return False
        except Exception as e:
            logger.error(f"Error getting use_margin setting: {e}")
            return False

    def _format_response(self, success: bool, message: str, results: Dict) -> Dict:
        """
        Format Lambda response

        Args:
            success: Whether execution was successful
            message: Status message
            results: Execution results

        Returns:
            Lambda response dict
        """
        return {
            'statusCode': 200 if success else 500,
            'body': json.dumps({
                'success': success,
                'message': message,
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            })
        }

    def _check_end_of_day_exit(
        self,
        services: Dict,
        current_positions: Dict,
        portfolio_value: float,
        pdt_status: Dict,
        balance_tracker,
        use_margin: bool
    ) -> Dict:
        """
        Check and execute end-of-day exits with PDT-aware logic

        Returns dict with:
            - eod_exit_triggered: bool
            - positions_exited: int
            - positions_held_overnight: int
            - exit_details: list
        """
        from datetime import timezone
        import pytz

        result = {
            'eod_exit_triggered': False,
            'positions_exited': 0,
            'positions_held_overnight': 0,
            'exit_details': []
        }

        try:
            config = services['config']
            intraday_exit_config = config.get('intraday_exit_rules', {})

            # Check if EOD exits enabled
            if not intraday_exit_config.get('eod_exit_enabled', True):
                return result

            # Check current time vs EOD exit time
            et = pytz.timezone('America/New_York')
            current_time_et = datetime.now(et)
            eod_hour = intraday_exit_config.get('eod_exit_hour', 15)  # 3 PM
            eod_minute = intraday_exit_config.get('eod_exit_minute', 45)  # 45 min

            eod_time = current_time_et.replace(hour=eod_hour, minute=eod_minute, second=0, microsecond=0)

            if current_time_et < eod_time:
                logger.debug(f"Before EOD exit time ({eod_hour}:{eod_minute:02d} ET)")
                return result

            logger.info(f"⏰ EOD EXIT TIME REACHED ({eod_hour}:{eod_minute:02d} ET) - Evaluating positions for exit")

            if not current_positions:
                logger.info("No positions to exit")
                return result

            # PDT-aware exit logic
            pdt_aware = intraday_exit_config.get('pdt_aware_mode', False)
            account_value = pdt_status.account_value if pdt_status else portfolio_value
            day_trades_remaining = pdt_status.remaining_day_trades if pdt_status else 3
            is_pdt_restricted = account_value < 25000

            # Scenario 1: Account >= $25K (PDT exempt)
            if not is_pdt_restricted:
                logger.info(f"💰 Account >= $25K (${account_value:,.2f}) - PDT EXEMPT, exiting all positions")
                exit_all = True

            # Scenario 2: PDT restricted with NO day trades remaining
            elif pdt_aware and is_pdt_restricted and day_trades_remaining == 0:
                logger.warning(
                    f"⚠️  NO DAY TRADES REMAINING (0/3) - HOLDING ALL {len(current_positions)} POSITIONS OVERNIGHT! "
                    f"Account: ${account_value:,.2f}"
                )
                result['positions_held_overnight'] = len(current_positions)
                for symbol in current_positions.keys():
                    logger.info(f"  🌙 Holding {symbol} overnight (no day trades)")
                return result  # Exit early, hold all positions

            # Scenario 3: PDT restricted with day trades available
            elif pdt_aware and is_pdt_restricted and day_trades_remaining > 0:
                logger.info(
                    f"⚠️  {day_trades_remaining} DAY TRADE(S) REMAINING ({3-day_trades_remaining}/3 used) - "
                    f"SELECTIVE EOD EXITS for significant losses only"
                )
                exit_all = False

                # Get loss threshold
                loss_threshold = intraday_exit_config.get('pdt_loss_exit_threshold', -2.0)

                # Calculate P&L for each position
                positions_with_pl = []
                for symbol, position in current_positions.items():
                    entry_price = safe_float(position.get('avg_entry_price', 0))
                    current_price = safe_float(position.get('current_price', 0))
                    quantity = safe_float(position.get('quantity', 0))

                    if entry_price > 0:
                        unrealized_pl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        unrealized_pl_pct = 0

                    positions_with_pl.append({
                        'symbol': symbol,
                        'pl_pct': unrealized_pl_pct,
                        'position': position
                    })

                # Sort by P/L (worst losses first)
                positions_with_pl.sort(key=lambda x: x['pl_pct'])

                # Identify significant losses
                significant_losses = [p for p in positions_with_pl if p['pl_pct'] < loss_threshold]

                if not significant_losses:
                    logger.info(f"✅ No positions with losses >{abs(loss_threshold)}%, holding all overnight")
                    result['positions_held_overnight'] = len(current_positions)
                    return result

                # Limit exits to available day trades
                positions_to_exit = significant_losses[:day_trades_remaining]
                positions_to_hold = len(current_positions) - len(positions_to_exit)

                logger.info(
                    f"📊 Exiting {len(positions_to_exit)} worst position(s) with losses >{abs(loss_threshold)}%, "
                    f"holding {positions_to_hold} overnight"
                )

                # Execute selective exits
                for pos_data in positions_to_exit:
                    symbol = pos_data['symbol']
                    pl_pct = pos_data['pl_pct']
                    position = pos_data['position']

                    logger.info(f"  🔻 Exiting {symbol} (EOD loss exit: {pl_pct:+.2f}%)")

                    # Execute sell
                    sell_result = self._execute_simple_sell(
                        services, symbol, position, 'EOD_PDT_LOSS_EXIT', use_margin
                    )

                    if sell_result['executed']:
                        result['positions_exited'] += 1
                        result['exit_details'].append({
                            'symbol': symbol,
                            'reason': 'EOD_PDT_LOSS_EXIT',
                            'pl_pct': pl_pct
                        })

                        # Update balance tracker
                        quantity_sold = sell_result.get('quantity', position['quantity'])
                        if use_margin:
                            balance_tracker.update_after_sell(sell_result['proceeds'], symbol, quantity_sold)
                        else:
                            balance_tracker.update_after_sell(sell_result['proceeds'], symbol, quantity_sold)

                result['positions_held_overnight'] = positions_to_hold
                result['eod_exit_triggered'] = True

                # Log held positions
                held_symbols = [p['symbol'] for p in positions_with_pl if p not in positions_to_exit]
                for symbol in held_symbols:
                    logger.info(f"  🌙 Holding {symbol} overnight")

                return result

            else:
                # Standard EOD exit (not PDT aware or other scenario)
                exit_all = True

            # Exit all positions
            if exit_all:
                logger.info(f"Exiting all {len(current_positions)} positions at EOD")

                for symbol, position in current_positions.items():
                    logger.info(f"  Exiting {symbol} (EOD exit)")

                    sell_result = self._execute_simple_sell(
                        services, symbol, position, 'EOD_EXIT', use_margin
                    )

                    if sell_result['executed']:
                        result['positions_exited'] += 1
                        result['exit_details'].append({
                            'symbol': symbol,
                            'reason': 'EOD_EXIT'
                        })

                        # Update balance tracker
                        quantity_sold = sell_result.get('quantity', position['quantity'])
                        if use_margin:
                            balance_tracker.update_after_sell(sell_result['proceeds'], symbol, quantity_sold)
                        else:
                            balance_tracker.update_after_sell(sell_result['proceeds'], symbol, quantity_sold)

                result['eod_exit_triggered'] = True

            logger.info(f"EOD exit complete: {result['positions_exited']} exited, {result['positions_held_overnight']} held overnight")
            return result

        except Exception as e:
            logger.error(f"Error in EOD exit check: {e}")
            return result

    def _execute_simple_sell(
        self,
        services: Dict,
        symbol: str,
        position: Dict,
        exit_reason: str,
        use_margin: bool
    ) -> Dict:
        """
        Execute a simple market sell (used for exits, stops, take-profits)

        Returns dict with:
            - executed: bool
            - proceeds: float (if executed)
            - quantity: float
            - price: float
        """
        result = {'executed': False, 'proceeds': 0.0, 'quantity': 0.0, 'price': 0.0}

        try:
            order_service = services['order_service']
            alpaca_client = services['alpaca_client']

            quantity = safe_float(position.get('quantity', 0))
            current_price = safe_float(position.get('current_price', 0))

            if quantity <= 0:
                logger.warning(f"Invalid quantity for {symbol}: {quantity}")
                return result

            # Execute market sell
            order_result = alpaca_client.place_market_order_with_confirmation(
                symbol=symbol,
                qty=quantity,
                side='sell',
                expected_price=current_price
            )

            if order_result.is_success:
                filled_price = safe_float(order_result.filled_avg_price, default=current_price)
                filled_qty = safe_float(order_result.filled_quantity, default=quantity)
                proceeds = filled_price * filled_qty

                # Create trade record
                import uuid

                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    symbol=symbol,
                    side='sell',
                    quantity=filled_qty,
                    price=filled_price,
                    order_type='market',
                    timestamp=datetime.utcnow().isoformat(),
                    strategy_signals={},  # Sell signal, not from strategy analysis
                    consensus_score=0.0,  # N/A for exit trades
                    risk_checks_passed=True,  # Exit trade, already validated
                    risk_check_details={'exit_reason': exit_reason},
                    exit_reason=exit_reason,
                    status='filled'
                )

                # Save trade
                order_service.save_trade(trade)

                # Calculate and save tax obligation
                order_service._calculate_and_save_tax(
                    symbol=symbol,
                    sell_quantity=filled_qty,
                    sell_price=filled_price,
                    trade_id=trade.trade_id
                )

                result['executed'] = True
                result['proceeds'] = proceeds
                result['quantity'] = filled_qty
                result['price'] = filled_price

                logger.info(f"  ✓ Sold {filled_qty} {symbol} @ ${filled_price:.2f} ({exit_reason})")

            else:
                logger.warning(f"  ✗ Sell order not filled for {symbol}: {order_result.status.value}")

            return result

        except Exception as e:
            logger.error(f"Error executing sell for {symbol}: {e}")
            return result

    def _execute_partial_sell(
        self,
        services: Dict,
        symbol: str,
        position: Dict,
        quantity_to_sell: int,
        exit_reason: str,
        use_margin: bool
    ) -> Dict:
        """
        Execute a partial market sell (used for trimming oversized positions)

        Args:
            quantity_to_sell: Specific number of shares to sell (not full position)

        Returns dict with:
            - executed: bool
            - proceeds: float (if executed)
            - quantity: float
            - price: float
        """
        result = {'executed': False, 'proceeds': 0.0, 'quantity': 0.0, 'price': 0.0}

        try:
            order_service = services['order_service']
            alpaca_client = services['alpaca_client']

            current_price = safe_float(position.get('current_price', 0))

            if quantity_to_sell <= 0:
                logger.warning(f"Invalid quantity to sell for {symbol}: {quantity_to_sell}")
                return result

            # Execute market sell for specific quantity
            order_result = alpaca_client.place_market_order_with_confirmation(
                symbol=symbol,
                qty=quantity_to_sell,
                side='sell',
                expected_price=current_price
            )

            if order_result.is_success:
                filled_price = safe_float(order_result.filled_avg_price, default=current_price)
                filled_qty = safe_float(order_result.filled_quantity, default=quantity_to_sell)
                proceeds = filled_price * filled_qty

                # Create trade record
                import uuid

                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    symbol=symbol,
                    side='sell',
                    quantity=filled_qty,
                    price=filled_price,
                    order_type='market',
                    timestamp=datetime.utcnow().isoformat(),
                    strategy_signals={},
                    consensus_score=0.0,
                    risk_checks_passed=True,
                    risk_check_details={'exit_reason': exit_reason, 'partial_sell': True},
                    exit_reason=exit_reason,
                    status='filled'
                )

                # Save trade
                order_service.save_trade(trade)

                # Calculate and save tax obligation
                order_service._calculate_and_save_tax(
                    symbol=symbol,
                    sell_quantity=filled_qty,
                    sell_price=filled_price,
                    trade_id=trade.trade_id
                )

                result['executed'] = True
                result['proceeds'] = proceeds
                result['quantity'] = filled_qty
                result['price'] = filled_price

                logger.info(f"  ✓ Partial sell: {filled_qty} {symbol} @ ${filled_price:.2f} ({exit_reason})")

            else:
                logger.warning(f"  ✗ Partial sell order not filled for {symbol}: {order_result.status.value}")

            return result

        except Exception as e:
            logger.error(f"Error executing partial sell for {symbol}: {e}")
            return result

    def _check_stop_losses(
        self,
        services: Dict,
        current_positions: Dict,
        portfolio_value: float,
        balance_tracker,
        use_margin: bool,
        pdt_growth_mode: bool
    ) -> Dict:
        """
        Check and execute stop-loss exits (static, trailing, breakeven)

        Returns dict with:
            - stop_losses_triggered: int
            - exit_details: list
        """
        result = {'stop_losses_triggered': 0, 'exit_details': []}

        try:
            dynamodb_client = services['dynamodb_client']
            positions_table_name = self.positions_table_name

            # Get positions from DynamoDB (has stop data)
            positions_table = dynamodb_client.Table(positions_table_name)
            response = positions_table.scan()
            db_positions = {item['symbol']: item for item in response.get('Items', [])}

            if not db_positions:
                logger.debug("No DynamoDB positions with stop data")
                return result

            for symbol, db_position in db_positions.items():
                if symbol not in current_positions:
                    continue  # Position no longer held

                position = current_positions[symbol]
                stop_loss_price = safe_float(db_position.get('stop_loss_price', 0))
                current_price = safe_float(position.get('current_price', 0))
                entry_price = safe_float(db_position.get('avg_entry_price', 0))

                if stop_loss_price <= 0 or current_price <= 0 or entry_price <= 0:
                    continue

                # Calculate profit %
                profit_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

                # PDT-AWARE DYNAMIC STOP-LOSS LOGIC
                # Check if selling this position would trigger a day trade
                pdt_aware_stops = services['config'].get('execution', {}).get('pdt_aware_stops', False)

                if pdt_aware_stops:
                    from datetime import datetime, timezone
                    entry_timestamp_str = db_position.get('entry_timestamp', '')

                    if entry_timestamp_str:
                        try:
                            entry_date = datetime.fromisoformat(entry_timestamp_str.replace('Z', '+00:00')).date()
                            today = datetime.now(timezone.utc).date()
                            is_same_day_position = (entry_date == today)

                            if is_same_day_position:
                                # Position bought today - selling would be a day trade
                                # Keep original wide stop (no adjustment)
                                logger.debug(f"  💎 {symbol}: Keeping wide stop (bought today, PDT protection)")
                            else:
                                # Position held overnight - can sell without day trade
                                # Aggressively tighten stop to protect capital
                                breakeven_on_profit = services['config'].get('execution', {}).get('breakeven_on_overnight_profit', True)
                                overnight_trailing_pct = services['config'].get('execution', {}).get('overnight_loss_trailing_percent', 0.10)

                                # Option 1: Move to breakeven if profitable
                                if profit_pct > 0 and stop_loss_price < entry_price and breakeven_on_profit:
                                    new_stop = entry_price
                                    logger.info(f"  🛡️ {symbol}: PDT-aware stop → BREAKEVEN (overnight position, P/L: {profit_pct:+.2f}%)")

                                    positions_table.update_item(
                                        Key={'symbol': symbol},
                                        UpdateExpression='SET stop_loss_price = :stop',
                                        ExpressionAttributeValues={':stop': Decimal(str(new_stop))}
                                    )
                                    stop_loss_price = new_stop

                                # Option 2: Tight trailing stop for losses
                                elif profit_pct < 0:
                                    new_stop = current_price * (1 - overnight_trailing_pct)  # Default: 10% below current
                                    if new_stop > stop_loss_price:
                                        logger.info(f"  ⚡ {symbol}: PDT-aware stop → TIGHT (${new_stop:.2f}, overnight loss, {profit_pct:.2f}%)")

                                        positions_table.update_item(
                                            Key={'symbol': symbol},
                                            UpdateExpression='SET stop_loss_price = :stop',
                                            ExpressionAttributeValues={':stop': Decimal(str(new_stop))}
                                        )
                                        stop_loss_price = new_stop
                        except Exception as e:
                            logger.warning(f"Error parsing entry timestamp for {symbol}: {e}")

                # Breakeven stop logic (only if not PDT growth mode)
                if not pdt_growth_mode:
                    trailing_stop_active = db_position.get('trailing_stop_active', False)
                    intraday_enabled = services['config'].get('intraday_exit_rules', {}).get('enable_intraday_stops', True)

                    if intraday_enabled and not trailing_stop_active and profit_pct >= 1.5:
                        # Move stop to breakeven
                        new_stop = entry_price
                        logger.info(f"  🔄 {symbol}: Moving stop to breakeven (profit: {profit_pct:.2f}%)")

                        # Update DynamoDB
                        positions_table.update_item(
                            Key={'symbol': symbol},
                            UpdateExpression='SET stop_loss_price = :stop',
                            ExpressionAttributeValues={':stop': Decimal(str(new_stop))}
                        )
                        stop_loss_price = new_stop

                    # Trailing stop logic (only if not PDT growth mode)
                    if trailing_stop_active and not pdt_growth_mode:
                        peak_price = safe_float(db_position.get('peak_price', entry_price))

                        # Update peak if new high
                        if current_price > peak_price:
                            peak_price = current_price
                            gain_from_entry = current_price - entry_price
                            new_trailing_stop = entry_price + (gain_from_entry * 0.5)  # Keep 50% of gain

                            logger.info(f"  ⬆️  {symbol}: New peak ${current_price:.2f}, trailing stop → ${new_trailing_stop:.2f}")

                            # Update DynamoDB
                            updates = convert_floats_to_decimal({
                                'peak_price': peak_price,
                                'stop_loss_price': new_trailing_stop
                            })

                            positions_table.update_item(
                                Key={'symbol': symbol},
                                UpdateExpression='SET peak_price = :peak, stop_loss_price = :stop',
                                ExpressionAttributeValues={
                                    ':peak': updates['peak_price'],
                                    ':stop': updates['stop_loss_price']
                                }
                            )
                            stop_loss_price = new_trailing_stop

                # Check if stop breached
                if current_price <= stop_loss_price:
                    loss_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

                    logger.warning(f"  🛑 STOP LOSS TRIGGERED: {symbol} @ ${current_price:.2f} (stop: ${stop_loss_price:.2f}, {loss_pct:+.2f}%)")

                    # Execute sell
                    sell_result = self._execute_simple_sell(
                        services, symbol, position, 'STOP_LOSS', use_margin
                    )

                    if sell_result['executed']:
                        result['stop_losses_triggered'] += 1
                        result['exit_details'].append({
                            'symbol': symbol,
                            'reason': 'STOP_LOSS',
                            'exit_price': sell_result['price'],
                            'stop_price': stop_loss_price,
                            'pl_pct': loss_pct
                        })

                        # Update balance tracker
                        quantity_sold = sell_result.get('quantity', position.get('quantity', 0))
                        balance_tracker.update_after_sell(sell_result['proceeds'], symbol, quantity_sold)

                        # Delete from DynamoDB
                        positions_table.delete_item(Key={'symbol': symbol})

            if result['stop_losses_triggered'] > 0:
                logger.info(f"Stop losses: {result['stop_losses_triggered']} position(s) exited")

            return result

        except Exception as e:
            logger.error(f"Error checking stop losses: {e}")
            return result

    def _check_take_profit_targets(
        self,
        services: Dict,
        current_positions: Dict,
        portfolio_value: float,
        balance_tracker,
        use_margin: bool,
        pdt_growth_mode: bool
    ) -> Dict:
        """
        Check and execute take-profit targets (3-level progressive ladder)

        Returns dict with:
            - take_profits_triggered: int
            - exit_details: list
        """
        result = {'take_profits_triggered': 0, 'exit_details': []}

        try:
            config = services['config']
            intraday_exit_config = config.get('intraday_exit_rules', {})

            # Check if enabled
            if not intraday_exit_config.get('enable_take_profit', True):
                logger.debug("Take-profit disabled in config")
                return result

            # DISABLED in PDT growth mode
            if pdt_growth_mode:
                logger.debug("💎 PDT GROWTH MODE: Take-profit DISABLED - letting winners run")
                return result

            # Get settings
            initial_tp = intraday_exit_config.get('initial_take_profit_pct', 6.5)
            tp_increment = intraday_exit_config.get('take_profit_increment_pct', 3.5)
            partial_exit_pct = intraday_exit_config.get('partial_exit_pct', 33)
            max_levels = intraday_exit_config.get('max_take_profit_levels', 3)

            dynamodb_client = services['dynamodb_client']
            positions_table_name = self.positions_table_name
            positions_table = dynamodb_client.Table(positions_table_name)

            # Get positions from DynamoDB
            response = positions_table.scan()
            db_positions = {item['symbol']: item for item in response.get('Items', [])}

            for symbol, db_position in db_positions.items():
                if symbol not in current_positions:
                    continue

                position = current_positions[symbol]
                entry_price = safe_float(db_position.get('avg_entry_price', 0))
                current_price = safe_float(position.get('current_price', 0))
                quantity = safe_float(position.get('quantity', 0))

                if entry_price <= 0 or current_price <= 0 or quantity <= 0:
                    continue

                # Calculate profit %
                profit_pct = ((current_price - entry_price) / entry_price) * 100

                # Get current take-profit level
                current_tp_level = safe_int(db_position.get('take_profit_level', 0))

                if current_tp_level >= max_levels:
                    continue  # Already at max levels

                # Calculate next target
                next_target_pct = initial_tp + (current_tp_level * tp_increment)
                # Level 0 → 6.5%, Level 1 → 10%, Level 2 → 13.5%

                if profit_pct >= next_target_pct:
                    # Calculate exit quantity (partial)
                    exit_quantity = quantity * (partial_exit_pct / 100.0)
                    exit_quantity = max(1, int(exit_quantity))  # At least 1 share

                    logger.info(
                        f"  💰 TAKE-PROFIT L{current_tp_level + 1}: {symbol} @ ${current_price:.2f} "
                        f"(target: {next_target_pct:.1f}%, actual: {profit_pct:.2f}%) - "
                        f"Exiting {exit_quantity}/{quantity} shares"
                    )

                    # Create partial position for sell
                    partial_position = position.copy()
                    partial_position['quantity'] = exit_quantity

                    # Execute partial sell
                    sell_result = self._execute_simple_sell(
                        services, symbol, partial_position, f'TAKE_PROFIT_L{current_tp_level + 1}', use_margin
                    )

                    if sell_result['executed']:
                        result['take_profits_triggered'] += 1
                        result['exit_details'].append({
                            'symbol': symbol,
                            'reason': f'TAKE_PROFIT_L{current_tp_level + 1}',
                            'exit_price': sell_result['price'],
                            'quantity': exit_quantity,
                            'profit_pct': profit_pct
                        })

                        # Update balance tracker
                        quantity_sold = sell_result.get('quantity', position.get('quantity', 0))
                        balance_tracker.update_after_sell(sell_result['proceeds'], symbol, quantity_sold)

                        # Update DynamoDB
                        new_quantity = quantity - exit_quantity

                        if new_quantity <= 0:
                            # Fully exited
                            positions_table.delete_item(Key={'symbol': symbol})
                            logger.info(f"    Position fully exited")
                        else:
                            # Update remaining quantity and increment TP level
                            updates = convert_floats_to_decimal({
                                'quantity': new_quantity,
                                'take_profit_level': current_tp_level + 1
                            })

                            # Activate trailing stop after first take-profit
                            if current_tp_level == 0:
                                updates['trailing_stop_active'] = True
                                updates['peak_price'] = current_price
                                logger.info(f"    Trailing stop ACTIVATED for remaining {new_quantity} shares")

                            positions_table.update_item(
                                Key={'symbol': symbol},
                                UpdateExpression='SET quantity = :qty, take_profit_level = :level, trailing_stop_active = :ts, peak_price = :peak',
                                ExpressionAttributeValues={
                                    ':qty': updates['quantity'],
                                    ':level': updates['take_profit_level'],
                                    ':ts': updates.get('trailing_stop_active', False),
                                    ':peak': updates.get('peak_price', Decimal('0'))
                                }
                            )

            if result['take_profits_triggered'] > 0:
                logger.info(f"Take-profits: {result['take_profits_triggered']} partial exit(s)")

            return result

        except Exception as e:
            logger.error(f"Error checking take-profit targets: {e}")
            return result

    def _fetch_historical_data(
        self,
        market_data_service: MarketDataService,
        trading_universe: List[str],
        current_positions: Dict
    ) -> Dict:
        """
        Fetch historical daily bars for trading universe + SPY + positions

        Args:
            market_data_service: Market data service instance
            trading_universe: List of symbols to analyze
            current_positions: Current positions dictionary

        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        try:
            # Combine universe, SPY, and position symbols
            symbols_to_fetch = set(trading_universe)
            symbols_to_fetch.add('SPY')  # Always include SPY benchmark

            # Include all held positions
            for symbol in current_positions.keys():
                symbols_to_fetch.add(symbol)

            symbols_list = list(symbols_to_fetch)
            logger.info(f"Fetching historical data for {len(symbols_list)} symbols (universe + SPY + positions)")

            # Fetch 300 calendar days to ensure 200+ trading days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=300)

            historical_data = market_data_service.get_historical_data(
                symbols=symbols_list,
                start=start_date,
                end=end_date,
                timeframe=TimeFrame.Day
            )

            logger.info(f"Successfully fetched historical data for {len(historical_data)} symbols")
            return historical_data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return {}

    def _fetch_intraday_data(
        self,
        market_data_service: MarketDataService,
        trading_universe: List[str],
        current_positions: Dict,
        config: Dict
    ) -> Dict:
        """
        Fetch intraday 5-minute bars with caching

        Args:
            market_data_service: Market data service instance
            trading_universe: List of symbols to analyze
            current_positions: Current positions dictionary
            config: Configuration dictionary

        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        try:
            # Check if market is open
            if not market_data_service.alpaca_client.is_market_open():
                logger.info("Market not yet open, skipping intraday data fetch")
                return {}

            # Combine universe, SPY, and position symbols
            symbols_to_fetch = set(trading_universe)
            symbols_to_fetch.add('SPY')

            for symbol in current_positions.keys():
                symbols_to_fetch.add(symbol)

            symbols_list = list(symbols_to_fetch)

            # Get intraday data with caching (use timezone-aware datetimes)
            from datetime import timezone
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=2)  # Last 2 hours of intraday data
            intraday_data = market_data_service.get_cached_intraday_data(
                symbols=symbols_list,
                start=start_time,
                end=end_time
            )

            if intraday_data:
                logger.info(f"Successfully fetched intraday data for {len(intraday_data)} symbols")
            else:
                logger.info("No intraday data available yet")

            return intraday_data

        except Exception as e:
            logger.error(f"Error fetching intraday data: {e}")
            return {}

    def _apply_pre_filter(
        self,
        market_scanner: MarketScanner,
        trading_universe: List[str],
        historical_data: Dict,
        intraday_data: Dict
    ) -> List[str]:
        """
        Apply pre-filter to reduce universe before expensive analysis

        Args:
            market_scanner: Market scanner instance
            trading_universe: List of symbols
            historical_data: Historical data dictionary
            intraday_data: Intraday data dictionary

        Returns:
            Filtered list of symbols
        """
        try:
            logger.info(f"Pre-filter: Starting with {len(trading_universe)} symbols")

            filtered_universe, filter_stats = market_scanner.pre_filter_candidates(
                symbols=trading_universe,
                historical_data=historical_data,
                intraday_data=intraday_data
            )

            pct_filtered = ((len(trading_universe) - len(filtered_universe)) / len(trading_universe) * 100) if trading_universe else 0
            logger.info(f"Pre-filter: Reduced to {len(filtered_universe)} symbols ({pct_filtered:.1f}% filtered)")

            return filtered_universe

        except Exception as e:
            logger.error(f"Error applying pre-filter: {e}")
            return trading_universe

    def _ensure_positions_in_universe(
        self,
        trading_universe: List[str],
        current_positions: Dict
    ) -> List[str]:
        """
        Ensure all held positions are included in trading universe

        Args:
            trading_universe: Current trading universe
            current_positions: Current positions dictionary

        Returns:
            Updated trading universe including all positions
        """
        universe_set = set(trading_universe)
        original_size = len(universe_set)

        # Add all held positions
        for symbol in current_positions.keys():
            universe_set.add(symbol)

        if len(universe_set) > original_size:
            added = len(universe_set) - original_size
            logger.info(f"Added {added} held position(s) to trading universe")

        return list(universe_set)

    def _execute_buy(
        self,
        services: Dict,
        symbol: str,
        historical_data: Dict,
        portfolio_value: float,
        current_positions: Dict,
        consensus_score: float,
        strategy_signals: Dict,
        balance_tracker,
        use_margin: bool,
        pdt_status: Dict,
        total_buy_signals: int = 1
    ) -> Dict:
        """
        Execute a buy order with full validation and rebalancing

        Args:
            total_buy_signals: Number of buy signals in this execution cycle.
                              Used for position sizing decisions.

        Returns dict with:
            - executed: bool
            - quantity: float
            - price: float
            - reason: str (if rejected)
        """
        result = {'executed': False, 'quantity': 0, 'price': 0.0, 'reason': ''}

        try:
            config = services['config']
            risk_manager = services['risk_manager']
            alpaca_client = services['alpaca_client']
            order_service = services['order_service']
            dynamodb_client = services['dynamodb_client']

            # Get current price
            data = historical_data.get(symbol)
            if data is None or data.empty:
                result['reason'] = 'No historical data'
                return result

            current_price = safe_float(data['close'].iloc[-1])
            if current_price <= 0:
                result['reason'] = 'Invalid price'
                return result

            # 1. WASH SALE CHECK (first priority) - only if enabled
            wash_sale_tracker_instance = services.get('wash_sale_tracker')
            if wash_sale_tracker_instance:
                # Get closed trades for this symbol to check wash sale
                trades_table = dynamodb_client.Table(self.trades_table_name)
                try:
                    response = trades_table.query(
                        IndexName='symbol-timestamp-index',
                        KeyConditionExpression='symbol = :symbol',
                        ExpressionAttributeValues={':symbol': symbol}
                    )
                    closed_trades = [t for t in response.get('Items', []) if t.get('action') == 'SELL']
                    wash_sale_check = wash_sale_tracker_instance.check_wash_sale(symbol, 'buy', closed_trades)

                    if not wash_sale_check.can_buy:
                        logger.warning(f"  ⚠️  WASH SALE RULE: Blocking buy of {symbol} (sold at loss within 30 days)")
                        result['reason'] = 'Wash sale rule'
                        self._save_rejected_trade(services, symbol, current_price, 0, 'WASH_SALE', consensus_score)
                        return result
                except Exception as e:
                    logger.warning(f"Error checking wash sale for {symbol}: {e}. Proceeding with trade.")

            # 2. POSITION SIZING
            atr_series = TechnicalIndicators.atr(data['high'], data['low'], data['close'])
            if atr_series.empty or len(atr_series) == 0:
                result['reason'] = 'Insufficient data for ATR'
                return result
            atr = atr_series.iloc[-1]
            if pd.isna(atr) or atr <= 0:
                result['reason'] = 'Invalid ATR'
                return result

            quantity = risk_manager.calculate_position_size(
                symbol=symbol,
                price=current_price,
                portfolio_value=portfolio_value,
                atr=atr
            )

            if quantity <= 0:
                result['reason'] = 'Position size = 0'
                return result

            # 3. CREATE TRADE OBJECT
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=symbol,
                side='buy',
                quantity=quantity,
                price=current_price,
                order_type='market',
                timestamp=datetime.utcnow().isoformat(),
                consensus_score=consensus_score,
                strategy_signals=strategy_signals,
                risk_checks_passed=False,  # Will be updated after risk check
                risk_check_details={},  # Will be populated by risk manager
                status='pending'
            )

            # 4. PRELIMINARY RISK CHECK
            cash_available = balance_tracker.get_available_funds()
            cost = current_price * quantity

            risk_check = risk_manager.check_trade_risk(
                symbol=symbol,
                side='buy',
                quantity=quantity,
                price=current_price,
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                market_data=data,
                buying_power=balance_tracker.buying_power,
                cash_balance=balance_tracker.cash_balance
            )

            # Identify if ONLY cash/buying power is the issue
            has_cash_violation = any(
                'cash' in v.lower() or 'funds' in v.lower() or 'buying power' in v.lower()
                for v in risk_check.violations
            )
            cash_is_only_issue = (not risk_check.passed and
                                   has_cash_violation and
                                   cost > cash_available)

            # 5. INTELLIGENT REBALANCING (if needed)
            if cash_is_only_issue and current_positions:
                logger.info(f"  💰 Insufficient funds for {symbol} (need ${cost:,.2f}, have ${cash_available:,.2f}) - attempting rebalancing")

                position_evaluator = PositionEvaluator(
                    config=config.get('position_management', {})
                )

                should_rebalance, positions_to_sell = position_evaluator.evaluate_rebalance_decision(
                    candidate_symbol=symbol,
                    candidate_consensus=consensus_score,
                    current_positions=current_positions,
                    historical_data=historical_data,
                    cash_needed=cost
                )

                if should_rebalance and positions_to_sell:
                    logger.info(f"  🔄 Rebalancing: Selling {len(positions_to_sell)} weak position(s) to free ${cost:,.2f}")

                    freed_cash = 0
                    today = datetime.utcnow().date()

                    for pos_symbol in positions_to_sell:
                        if pos_symbol not in current_positions:
                            continue

                        position = current_positions[pos_symbol]

                        # PDT CHECK: Skip same-day positions to avoid day trade
                        entry_timestamp_str = position.get('entry_timestamp', '')
                        if entry_timestamp_str:
                            try:
                                entry_date = datetime.fromisoformat(
                                    entry_timestamp_str.replace('Z', '+00:00')
                                ).date()
                                if entry_date == today:
                                    logger.info(
                                        f"    ⏭️  Skipping {pos_symbol} - same-day position "
                                        f"(would trigger day trade)"
                                    )
                                    continue
                            except Exception as e:
                                logger.warning(f"    Could not parse entry_timestamp for {pos_symbol}: {e}")

                        logger.info(f"    Selling {pos_symbol} (weak position)")

                        sell_result = self._execute_simple_sell(
                            services, pos_symbol, position, 'REBALANCE', use_margin
                        )

                        if sell_result['executed']:
                            freed_cash += sell_result['proceeds']
                            quantity_sold = sell_result.get('quantity', position.get('quantity', 0))
                            balance_tracker.update_after_sell(sell_result['proceeds'], pos_symbol, quantity_sold)
                            del current_positions[pos_symbol]

                            if freed_cash >= cost:
                                break

                    if freed_cash >= cost:
                        logger.info(f"  ✅ Phase 1 Rebalancing successful: Freed ${freed_cash:,.2f}")
                        cash_available = balance_tracker.get_available_funds()
                    else:
                        # Phase 2: Trim oversized positions if still need more cash
                        remaining_needed = cost - freed_cash
                        logger.info(f"  📊 Phase 2: Trimming oversized positions (need ${remaining_needed:,.2f} more)")

                        risk_config = config.get('risk_management', {}).get('layer1_trade_level', {})
                        max_position_size = risk_config.get('max_position_size', 0.25)

                        for pos_symbol, position in list(current_positions.items()):
                            if remaining_needed <= 0:
                                break

                            if pos_symbol == symbol:
                                continue  # Don't trim the position we're trying to buy

                            # PDT CHECK: Skip same-day positions
                            entry_timestamp_str = position.get('entry_timestamp', '')
                            if entry_timestamp_str:
                                try:
                                    entry_date = datetime.fromisoformat(
                                        entry_timestamp_str.replace('Z', '+00:00')
                                    ).date()
                                    if entry_date == today:
                                        continue  # Skip same-day positions
                                except (ValueError, TypeError, AttributeError) as e:
                                    logger.debug(f"Could not parse entry_timestamp '{entry_timestamp_str}' for {pos_symbol}: {e}")

                            pos_value = position.get('market_value', position.get('quantity', 0) * position.get('current_price', 0))
                            position_pct = pos_value / portfolio_value if portfolio_value > 0 else 0

                            if position_pct <= max_position_size:
                                continue  # Not oversized

                            # Calculate how much to trim
                            target_value = portfolio_value * max_position_size
                            excess_value = pos_value - target_value
                            pos_price = position.get('current_price', 0)

                            if pos_price <= 0:
                                continue

                            trim_value = min(excess_value, remaining_needed)
                            trim_qty = int(trim_value / pos_price)

                            if trim_qty <= 0:
                                continue

                            logger.info(f"    ✂️  Trimming {pos_symbol}: {position_pct*100:.1f}% -> {max_position_size*100:.0f}% (selling {trim_qty} shares)")

                            # Execute partial sell
                            trim_result = self._execute_partial_sell(
                                services, pos_symbol, position, trim_qty, 'REBALANCE_TRIM', use_margin
                            )

                            if trim_result['executed']:
                                proceeds = trim_result['proceeds']
                                freed_cash += proceeds
                                remaining_needed -= proceeds
                                balance_tracker.update_after_sell(proceeds, pos_symbol, trim_qty)

                                # Update position quantity
                                new_qty = position.get('quantity', 0) - trim_qty
                                if new_qty > 0:
                                    current_positions[pos_symbol]['quantity'] = new_qty
                                    current_positions[pos_symbol]['market_value'] = new_qty * pos_price
                                else:
                                    del current_positions[pos_symbol]

                                logger.info(f"    ✅ Trimmed {trim_qty} shares for ${proceeds:,.2f}")

                        if freed_cash >= cost:
                            logger.info(f"  ✅ Rebalancing successful (Phase 1+2): Freed ${freed_cash:,.2f}")
                            cash_available = balance_tracker.get_available_funds()
                        else:
                            logger.warning(f"  ❌ Rebalancing insufficient: Only freed ${freed_cash:,.2f} of ${cost:,.2f} needed")
                            result['reason'] = 'Rebalancing insufficient'
                            self._save_rejected_trade(services, symbol, current_price, quantity, 'REBALANCE_INSUFFICIENT', consensus_score)
                            return result
                else:
                    result['reason'] = 'Cannot rebalance'
                    self._save_rejected_trade(services, symbol, current_price, quantity, 'CANNOT_REBALANCE', consensus_score)
                    return result

            # 6. FINAL RISK CHECK
            risk_check = risk_manager.check_trade_risk(
                symbol=symbol,
                side='buy',
                quantity=quantity,
                price=current_price,
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                market_data=data,
                buying_power=balance_tracker.buying_power,
                cash_balance=balance_tracker.cash_balance
            )

            if not risk_check.passed:
                reason = risk_check.violations[0] if risk_check.violations else 'Risk check failed'
                logger.warning(f"  ❌ Risk check failed for {symbol}: {reason}")
                result['reason'] = reason
                trade.status = 'rejected'
                trade.rejection_reason = reason
                order_service.save_trade(trade)
                return result

            # 7. PDT COMPLIANCE CHECK
            if pdt_status and pdt_status.is_pdt_restricted:
                pdt_tracker = PDTTracker(
                    pdt_threshold=pdt_status.pdt_threshold,
                    pdt_limit=pdt_status.day_trades_limit
                )

                # Get recent trades
                trades_table = dynamodb_client.Table(self.trades_table_name)
                lookback_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
                response = trades_table.scan(
                    FilterExpression='#ts >= :lookback',
                    ExpressionAttributeNames={'#ts': 'timestamp'},
                    ExpressionAttributeValues={':lookback': lookback_date}
                )
                recent_trades = response.get('Items', [])

                pdt_check = pdt_tracker.check_pdt_compliance(
                    symbol=symbol,
                    side='buy',
                    account_value=portfolio_value,
                    recent_trades=recent_trades,
                    current_positions=current_positions
                )

                if not pdt_check.can_trade:
                    logger.critical(f"  🚫 PDT VIOLATION RISK: {symbol} - {pdt_check.blocking_message}")
                    result['reason'] = pdt_check.blocking_message
                    self._save_rejected_trade(services, symbol, current_price, quantity, 'PDT_VIOLATION', consensus_score)
                    return result

            # 8. EXECUTE ORDER
            logger.info(f"  📈 Executing BUY: {quantity} {symbol} @ ${current_price:.2f} (cost: ${cost:,.2f})")

            order_result = alpaca_client.place_market_order_with_confirmation(
                symbol=symbol,
                qty=quantity,
                side='buy',
                expected_price=current_price
            )

            if not order_result.is_success:
                logger.warning(f"  ❌ Order not filled: {order_result.status.value}")
                result['reason'] = f'Order {order_result.status.value}: {order_result.error_message or order_result.rejection_reason}'
                trade.status = 'rejected'
                trade.rejection_reason = result['reason']
                order_service.save_trade(trade)
                return result

            # 9. HANDLE FILL
            filled_price = safe_float(order_result.filled_avg_price, default=current_price)
            filled_qty = safe_float(order_result.filled_quantity, default=quantity)
            actual_cost = filled_price * filled_qty

            # Check slippage
            slippage_pct = abs((filled_price - current_price) / current_price * 100) if current_price > 0 else 0
            slippage_tolerance = config.get('execution', {}).get('slippage_tolerance_pct', 1.5)

            if slippage_pct > slippage_tolerance:
                logger.error(f"  🚨 EXCESSIVE SLIPPAGE: {slippage_pct:.2f}% > {slippage_tolerance}% - REVERSING trade")

                # Reverse the trade
                reverse_result = alpaca_client.place_market_order_with_confirmation(
                    symbol=symbol,
                    qty=filled_qty,
                    side='sell',
                    expected_price=filled_price
                )

                result['reason'] = f'Excessive slippage ({slippage_pct:.2f}%)'
                self._save_rejected_trade(services, symbol, filled_price, filled_qty, 'EXCESSIVE_SLIPPAGE', consensus_score)
                return result

            # 10. SUCCESS - Update trade record
            trade.status = 'filled'
            trade.price = filled_price
            trade.quantity = filled_qty
            order_service.save_trade(trade)

            # 11. Calculate and store stop-loss
            atr_stop_multiplier = config.get('risk_management', {}).get('layer1_trade_level', {}).get('atr_stop_multiplier', 1.5)
            stop_loss_price = filled_price - (atr * atr_stop_multiplier)

            # 12. Save position to DynamoDB
            position_data = convert_floats_to_decimal({
                'symbol': symbol,
                'quantity': filled_qty,
                'avg_entry_price': filled_price,
                'stop_loss_price': stop_loss_price,
                'peak_price': filled_price,
                'trough_price': filled_price,
                'peak_price_pct': 0,  # 0% at entry
                'trough_price_pct': 0,  # 0% at entry
                'peak_unrealized_pl': 0,
                'trough_unrealized_pl': 0,
                'peak_unrealized_pl_pct': 0,
                'trough_unrealized_pl_pct': 0,
                'trailing_stop_active': False,
                'take_profit_level': 0,
                'entry_timestamp': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat()
            })

            positions_table = dynamodb_client.Table(self.positions_table_name)
            positions_table.put_item(Item=position_data)

            result['executed'] = True
            result['quantity'] = filled_qty
            result['price'] = filled_price
            result['stop_loss_price'] = stop_loss_price

            logger.info(f"  ✅ BUY FILLED: {filled_qty} {symbol} @ ${filled_price:.2f} (stop: ${stop_loss_price:.2f})")

            return result

        except Exception as e:
            logger.error(f"Error executing buy for {symbol}: {e}")
            result['reason'] = str(e)
            return result

    def _execute_sell(
        self,
        services: Dict,
        symbol: str,
        position: Dict,
        historical_data: Dict,
        portfolio_value: float,
        current_positions: Dict,
        consensus_score: float,
        strategy_signals: Dict,
        balance_tracker,
        use_margin: bool
    ) -> Dict:
        """
        Execute a sell order based on strategy signal

        Returns dict with:
            - executed: bool
            - quantity: float
            - price: float
            - pl: float
        """
        result = {'executed': False, 'quantity': 0, 'price': 0.0, 'pl': 0.0}

        try:
            risk_manager = services['risk_manager']
            alpaca_client = services['alpaca_client']
            order_service = services['order_service']

            quantity = safe_float(position.get('quantity', 0))
            current_price = safe_float(position.get('current_price', 0))
            entry_price = safe_float(position.get('avg_entry_price', 0))

            if quantity <= 0 or current_price <= 0:
                result['reason'] = 'Invalid position data'
                return result

            # Safety check: Verify position still exists in Alpaca before selling
            # This prevents race conditions where another Lambda already sold this position
            try:
                alpaca_position = alpaca_client.get_position(symbol)
                if alpaca_position is None:
                    logger.warning(f"Position {symbol} no longer exists in Alpaca - skipping sell")
                    result['reason'] = 'Position no longer exists'
                    return result
                # Use actual quantity from Alpaca in case it changed
                actual_qty = safe_float(alpaca_position.qty)
                if actual_qty != quantity:
                    logger.warning(f"Position {symbol} quantity changed: expected {quantity}, actual {actual_qty}")
                    quantity = actual_qty
                    if quantity <= 0:
                        result['reason'] = 'No quantity to sell'
                        return result
            except Exception as e:
                # If we can't verify, log warning but proceed with sell attempt
                # Alpaca will reject if position doesn't exist
                logger.warning(f"Could not verify position {symbol} before sell: {e}")

            # Get market data for this symbol
            data = historical_data.get(symbol)
            if data is None or data.empty:
                result['reason'] = 'No historical data'
                return result

            # Create trade object
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=symbol,
                side='sell',
                quantity=quantity,
                price=current_price,
                order_type='market',
                timestamp=datetime.utcnow().isoformat(),
                consensus_score=consensus_score,
                strategy_signals=strategy_signals,
                risk_checks_passed=False,  # Will be updated after risk check
                risk_check_details={},  # Will be populated by risk manager
                status='pending'
            )

            # Risk check (sells are higher priority, execute even with warnings)
            risk_check = risk_manager.check_trade_risk(
                symbol=symbol,
                side='sell',
                quantity=quantity,
                price=current_price,
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                market_data=data
            )

            if risk_check.warnings:
                logger.info(f"  ⚠️  Sell warnings for {symbol}: {risk_check.warnings}")

            # Execute sell
            logger.info(f"  📉 Executing SELL: {quantity} {symbol} @ ${current_price:.2f}")

            order_result = alpaca_client.place_market_order_with_confirmation(
                symbol=symbol,
                qty=quantity,
                side='sell',
                expected_price=current_price
            )

            if not order_result.is_success:
                logger.warning(f"  ❌ Sell order not filled: {order_result.status.value}")
                result['reason'] = f'Order {order_result.status.value}: {order_result.error_message or order_result.rejection_reason}'
                return result

            # Handle fill
            filled_price = safe_float(order_result.filled_avg_price, default=current_price)
            filled_qty = safe_float(order_result.filled_quantity, default=quantity)
            proceeds = filled_price * filled_qty

            # Calculate realized P/L
            cost_basis = entry_price * filled_qty
            realized_pl = proceeds - cost_basis
            realized_pl_pct = (realized_pl / cost_basis * 100) if cost_basis > 0 else 0

            # Check slippage (more tolerance for sells)
            slippage_pct = abs((filled_price - current_price) / current_price * 100) if current_price > 0 else 0
            if slippage_pct > 2.0:
                logger.warning(f"  ⚠️  Slippage on sell: {slippage_pct:.2f}%")

            # Update trade record
            trade.status = 'filled'
            trade.price = filled_price
            trade.quantity = filled_qty
            trade.realized_pl = realized_pl
            trade.realized_pl_pct = realized_pl_pct
            order_service.save_trade(trade)

            # Calculate and save tax obligation
            if hasattr(order_service, 'tax_service') and order_service.tax_service:
                order_service._calculate_and_save_tax(
                    symbol=symbol,
                    sell_quantity=filled_qty,
                    sell_price=filled_price,
                    trade_id=trade.trade_id
                )

            result['executed'] = True
            result['quantity'] = filled_qty
            result['price'] = filled_price
            result['pl'] = realized_pl

            logger.info(f"  ✅ SELL FILLED: {filled_qty} {symbol} @ ${filled_price:.2f} (P/L: ${realized_pl:+,.2f} / {realized_pl_pct:+.2f}%)")

            return result

        except Exception as e:
            logger.error(f"Error executing sell for {symbol}: {e}")
            result['reason'] = str(e)
            return result

    def _save_rejected_trade(
        self,
        services: Dict,
        symbol: str,
        price: float,
        quantity: float,
        reason: str,
        consensus_score: float
    ) -> None:
        """Save rejected trade to DynamoDB for analysis"""
        try:
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=symbol,
                side='buy',
                quantity=quantity,
                price=price,
                order_type='market',
                timestamp=datetime.utcnow().isoformat(),
                strategy_signals={},  # Not available for rejected trades
                consensus_score=consensus_score,
                risk_checks_passed=False,  # Rejected = failed risk checks
                risk_check_details={'rejection_reason': reason},
                status='rejected',
                rejection_reason=reason
            )

            order_service = services['order_service']
            order_service.save_trade(trade)

        except Exception as e:
            logger.error(f"Error saving rejected trade: {e}")

    def _update_balances_after_sells(
        self,
        alpaca_client: AlpacaClient,
        use_margin: bool
    ) -> Tuple[float, float]:
        """
        Update and return current cash balance and buying power after sells

        Args:
            alpaca_client: Alpaca client instance
            use_margin: Whether margin is enabled

        Returns:
            Tuple of (cash_balance, buying_power)
        """
        try:
            account = alpaca_client.get_account()
            cash_balance = safe_float(account.cash, default=0.0, field_name="cash_balance")
            buying_power = safe_float(account.buying_power, default=0.0, field_name="buying_power")

            if use_margin:
                logger.debug(f"Updated buying power: ${buying_power:,.2f}")
            else:
                logger.debug(f"Updated cash balance: ${cash_balance:,.2f}")

            return cash_balance, buying_power

        except Exception as e:
            logger.error(f"Error updating balances: {e}")
            return 0.0, 0.0

    def _refresh_positions(self, alpaca_client: AlpacaClient) -> Dict:
        """
        Refresh and return current positions from Alpaca

        Args:
            alpaca_client: Alpaca client instance

        Returns:
            Dictionary of current positions keyed by symbol
        """
        return self._get_current_positions(alpaca_client)

    def _update_risk_metrics(
        self,
        dynamodb_client,
        risk_metrics_table_name: str,
        portfolio_value: float,
        current_positions: Dict
    ) -> None:
        """
        Update risk metrics in DynamoDB

        Args:
            dynamodb_client: DynamoDB client
            risk_metrics_table_name: Name of risk metrics table
            portfolio_value: Current portfolio value
            current_positions: Current positions dictionary
        """
        try:
            risk_metrics = {
                'metric_type': 'portfolio',  # Partition key required by DynamoDB schema
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_value': portfolio_value,
                'num_positions': len(current_positions),
                'last_updated': datetime.utcnow().isoformat()
            }

            risk_metrics = convert_floats_to_decimal(risk_metrics)

            table = dynamodb_client.Table(risk_metrics_table_name)
            table.put_item(Item=risk_metrics)

            logger.debug("Updated risk metrics in DynamoDB")

        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")

    def _update_position_tracking(
        self,
        dynamodb_client,
        positions_table_name: str,
        current_positions: Dict
    ) -> None:
        """
        Update peak/trough tracking for all positions

        Args:
            dynamodb_client: DynamoDB client
            positions_table_name: Name of positions table
            current_positions: Current positions dictionary
        """
        if not current_positions:
            return

        try:
            positions_table = dynamodb_client.Table(positions_table_name)

            for symbol, position in current_positions.items():
                current_price = position.get('current_price', 0)
                avg_entry_price = position.get('avg_entry_price', 0)
                quantity = position.get('quantity', 0)

                if not current_price or not avg_entry_price or not quantity:
                    continue

                # Calculate current P&L
                cost_basis = avg_entry_price * quantity
                market_value = current_price * quantity
                unrealized_pl = market_value - cost_basis
                unrealized_pl_pct = (unrealized_pl / cost_basis) if cost_basis > 0 else 0

                # Get existing position data from DynamoDB
                try:
                    response = positions_table.get_item(Key={'symbol': symbol})
                    db_position = response.get('Item', {})
                except Exception:
                    db_position = {}

                # Get existing peak/trough values (or initialize from current)
                peak_price = float(db_position.get('peak_price', current_price))
                trough_price = float(db_position.get('trough_price', current_price))
                peak_unrealized_pl = float(db_position.get('peak_unrealized_pl', unrealized_pl))
                trough_unrealized_pl = float(db_position.get('trough_unrealized_pl', unrealized_pl))
                peak_unrealized_pl_pct = float(db_position.get('peak_unrealized_pl_pct', unrealized_pl_pct))
                trough_unrealized_pl_pct = float(db_position.get('trough_unrealized_pl_pct', unrealized_pl_pct))

                # Calculate price % change from entry
                price_pct_change = (current_price - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0

                # Update tracking values
                updates_needed = False
                updates = {}

                if current_price > peak_price:
                    updates['peak_price'] = Decimal(str(current_price))
                    updates['peak_price_pct'] = Decimal(str(price_pct_change))
                    updates_needed = True
                if current_price < trough_price:
                    updates['trough_price'] = Decimal(str(current_price))
                    updates['trough_price_pct'] = Decimal(str(price_pct_change))
                    updates_needed = True
                if unrealized_pl > peak_unrealized_pl:
                    updates['peak_unrealized_pl'] = Decimal(str(unrealized_pl))
                    updates['peak_unrealized_pl_pct'] = Decimal(str(unrealized_pl_pct))
                    updates_needed = True
                if unrealized_pl < trough_unrealized_pl:
                    updates['trough_unrealized_pl'] = Decimal(str(unrealized_pl))
                    updates['trough_unrealized_pl_pct'] = Decimal(str(unrealized_pl_pct))
                    updates_needed = True

                # Apply updates if any
                if updates_needed:
                    updates['last_updated'] = datetime.utcnow().isoformat()
                    update_expr = 'SET ' + ', '.join([f'{k} = :{k.replace("_", "")}' for k in updates.keys()])
                    expr_values = {f':{k.replace("_", "")}': v for k, v in updates.items()}

                    positions_table.update_item(
                        Key={'symbol': symbol},
                        UpdateExpression=update_expr,
                        ExpressionAttributeValues=expr_values
                    )
                    logger.debug(f"Updated tracking for {symbol}: peak=${peak_price:.2f}, trough=${trough_price:.2f}")

        except Exception as e:
            logger.error(f"Error updating position tracking: {e}")

    def _get_latest_risk_metrics(self, dynamodb_client, risk_metrics_table_name: str) -> Optional[Dict]:
        """
        Get latest risk metrics from DynamoDB

        Args:
            dynamodb_client: DynamoDB client
            risk_metrics_table_name: Name of risk metrics table

        Returns:
            Latest risk metrics or None
        """
        try:
            table = dynamodb_client.Table(risk_metrics_table_name)
            response = table.scan(Limit=1)
            items = response.get('Items', [])

            if items:
                return items[0]

            return None

        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return None

    def _validate_env_var(self, value: str, var_name: str) -> str:
        """
        Validate environment variable format to prevent injection issues

        Args:
            value: Environment variable value
            var_name: Name of the variable (for logging)

        Returns:
            Validated value

        Raises:
            ValueError: If value contains invalid characters
        """
        import re

        if not value:
            raise ValueError(f"{var_name} cannot be empty")

        # Allow only alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError(
                f"{var_name}='{value}' contains invalid characters. "
                f"Only alphanumeric, hyphens, and underscores are allowed."
            )

        logger.debug(f"Validated {var_name}: {value}")
        return value

    def _format_error_response(self, error_message: str) -> Dict:
        """
        Format error response

        Args:
            error_message: Error message

        Returns:
            Lambda error response
        """
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': error_message,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
