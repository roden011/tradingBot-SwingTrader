"""
Strategy Manager
Orchestrates multiple strategies and generates consensus signals

Supports both day trading and swing trading modes:
- Day Trading: Momentum Breakout, Opening Range Breakout, Relative Strength Intraday
- Swing Trading: Multi-Day Momentum, Position Builder, Relative Strength Intraday
"""
import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
from .momentum_breakout import MomentumBreakoutStrategy
from .opening_range_breakout import OpeningRangeBreakoutStrategy
from .relative_strength_intraday import RelativeStrengthIntradayStrategy
from .multi_day_momentum import MultiDayMomentumStrategy
from .position_builder import PositionBuilderStrategy
from .base_strategy import Signal

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages multiple trading strategies and generates consensus signals
    """

    def __init__(self, consensus_threshold: float = 0.30, config: dict = None):
        """
        Initialize strategy manager

        Args:
            consensus_threshold: Minimum weighted confidence to generate action (default 0.30)
            config: Full strategy configuration dict containing weights and strategy-specific params
        """
        self.consensus_threshold = consensus_threshold

        # Initialize strategy health tracking
        self.strategy_health = {
            'total_calls': 0,
            'strategy_stats': {},
            'strategies_skipped': 0
        }

        # Extract config sections
        config = config or {}
        weights = config.get('weights', {})
        strategy_configs = config

        # Conditional strategy execution settings
        self.conditional_strategies = config.get('conditional_strategies', True)

        # Initialize strategies based on weights provided in config
        self.strategies = []

        # Add strategies that have non-zero weights in config
        if 'momentum_breakout' in weights and weights['momentum_breakout'] > 0:
            self.strategies.append(MomentumBreakoutStrategy(
                weight=weights['momentum_breakout'],
                config=strategy_configs.get('momentum_breakout', {})
            ))

        if 'opening_range_breakout' in weights and weights['opening_range_breakout'] > 0:
            self.strategies.append(OpeningRangeBreakoutStrategy(
                weight=weights['opening_range_breakout'],
                config=strategy_configs.get('opening_range_breakout', {})
            ))

        if 'relative_strength_intraday' in weights and weights['relative_strength_intraday'] > 0:
            self.strategies.append(RelativeStrengthIntradayStrategy(
                weight=weights['relative_strength_intraday'],
                config=strategy_configs.get('relative_strength_intraday', {})
            ))

        if 'multi_day_momentum' in weights and weights['multi_day_momentum'] > 0:
            self.strategies.append(MultiDayMomentumStrategy(
                weight=weights['multi_day_momentum'],
                config=strategy_configs.get('multi_day_momentum', {})
            ))

        if 'position_builder' in weights and weights['position_builder'] > 0:
            self.strategies.append(PositionBuilderStrategy(
                weight=weights['position_builder'],
                config=strategy_configs.get('position_builder', {})
            ))

        # Verify weights sum to 1.0 (see archived_strategies.txt for old strategy reference)
        total_weight = sum(s.weight for s in self.strategies)
        assert abs(total_weight - 1.0) < 0.001, f"Strategy weights must sum to 1.0 (got {total_weight})"

        logger.info(
            f"Strategy manager initialized with {len(self.strategies)} strategies, "
            f"consensus threshold={consensus_threshold}, total_weight={total_weight:.3f}, "
            f"conditional_strategies={self.conditional_strategies}"
        )

    def _should_run_strategy(self, strategy_name: str, **kwargs) -> Tuple[bool, str]:
        """
        Check if a strategy should run based on time and market conditions

        Args:
            strategy_name: Name of the strategy
            **kwargs: Additional context (intraday_data, benchmark_data, etc.)

        Returns:
            Tuple of (should_run, skip_reason)
        """
        if not self.conditional_strategies:
            return True, ""

        # Get current ET time
        now_utc = datetime.now(timezone.utc)
        et_offset = timedelta(hours=5)
        now_et = now_utc - et_offset

        # Opening Range Breakout: Only run between 9:30 AM - 11:30 AM ET
        if strategy_name == "Opening Range Breakout":
            if now_et.hour > 11 or (now_et.hour == 11 and now_et.minute > 30):
                return False, "After 11:30 AM ET (ORB window closed)"
            elif now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30):
                return False, "Before 9:30 AM ET (market not open)"

        # Relative Strength Intraday: Skip if market movement < 0.5% in last hour
        elif strategy_name == "Relative Strength Intraday":
            spy_intraday_data = kwargs.get('spy_intraday_data')
            if spy_intraday_data is not None and not spy_intraday_data.empty:
                # Check SPY movement in last hour
                if len(spy_intraday_data) >= 12:  # Need at least 12 bars (1 hour of 5-min bars)
                    last_hour_start = spy_intraday_data['close'].iloc[-12]
                    last_price = spy_intraday_data['close'].iloc[-1]
                    movement_pct = abs((last_price - last_hour_start) / last_hour_start)

                    if movement_pct < 0.005:  # Less than 0.5%
                        return False, f"Low market movement ({movement_pct*100:.2f}% < 0.5%)"

        # Momentum Breakout: Always runs (can trigger anytime during market hours)

        return True, ""

    def generate_consensus_signal(
        self, symbol: str, data: pd.DataFrame, **kwargs
    ) -> Tuple[str, float, Dict]:
        """
        Generate consensus signal from all strategies

        Args:
            symbol: Stock symbol
            data: Historical price data
            **kwargs: Additional data (benchmark_data, etc.)

        Returns:
            Tuple of (action, consensus_score, strategy_signals)
            - action: 'buy', 'sell', or 'hold'
            - consensus_score: Weighted confidence score
            - strategy_signals: Dict of individual strategy signals
        """
        # Collect signals from all strategies
        strategy_signals = {}
        buy_score = 0.0
        sell_score = 0.0

        # Track this analysis call
        self.strategy_health['total_calls'] += 1

        for strategy in self.strategies:
            try:
                # Check if strategy should run
                should_run, skip_reason = self._should_run_strategy(strategy.name, **kwargs)

                if not should_run:
                    logger.debug(f"{strategy.name} ({symbol}): SKIPPED - {skip_reason}")
                    self.strategy_health['strategies_skipped'] += 1
                    strategy_signals[strategy.name] = {
                        'action': 'hold',
                        'confidence': 0.0,
                        'reason': f'Skipped: {skip_reason}',
                        'metadata': {'skipped': True},
                    }
                    continue

                signal = strategy.generate_signal(symbol, data, **kwargs)
                strategy_signals[strategy.name] = {
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'reason': signal.reason,
                    'metadata': signal.metadata,
                }

                # Track strategy health metrics
                if strategy.name not in self.strategy_health['strategy_stats']:
                    self.strategy_health['strategy_stats'][strategy.name] = {
                        'total_signals': 0,
                        'buy_signals': 0,
                        'sell_signals': 0,
                        'hold_signals': 0,
                        'error_count': 0,
                        'avg_confidence': 0.0,
                        'last_signal': None
                    }

                stats = self.strategy_health['strategy_stats'][strategy.name]
                stats['total_signals'] += 1
                stats['last_signal'] = signal.action

                # Update action counts
                if signal.action == 'buy':
                    stats['buy_signals'] += 1
                elif signal.action == 'sell':
                    stats['sell_signals'] += 1
                else:
                    stats['hold_signals'] += 1

                # Update average confidence (moving average)
                if signal.confidence > 0:
                    stats['avg_confidence'] = (stats['avg_confidence'] * (stats['total_signals'] - 1) +
                                               signal.confidence) / stats['total_signals']

                # Calculate weighted contribution
                weighted_confidence = signal.confidence * strategy.weight

                if signal.action == 'buy':
                    buy_score += weighted_confidence
                elif signal.action == 'sell':
                    sell_score += weighted_confidence

                logger.debug(
                    f"{strategy.name} ({symbol}): {signal.action} "
                    f"(confidence={signal.confidence:.2f}, weighted={weighted_confidence:.2f})"
                )

            except Exception as e:
                logger.error(f"Error in {strategy.name} for {symbol}: {e}")

                # Track error in health metrics
                if strategy.name not in self.strategy_health['strategy_stats']:
                    self.strategy_health['strategy_stats'][strategy.name] = {
                        'total_signals': 0,
                        'buy_signals': 0,
                        'sell_signals': 0,
                        'hold_signals': 0,
                        'error_count': 0,
                        'avg_confidence': 0.0,
                        'last_signal': None
                    }
                self.strategy_health['strategy_stats'][strategy.name]['error_count'] += 1

                strategy_signals[strategy.name] = {
                    'action': 'hold',
                    'confidence': 0.0,
                    'reason': f'Error: {str(e)}',
                    'metadata': {},
                }

        # Determine consensus action
        if buy_score >= self.consensus_threshold and buy_score > sell_score:
            action = 'buy'
            consensus_score = buy_score
        elif sell_score >= self.consensus_threshold and sell_score > buy_score:
            action = 'sell'
            consensus_score = sell_score
        else:
            action = 'hold'
            consensus_score = max(buy_score, sell_score)

        logger.info(
            f"Consensus for {symbol}: {action} "
            f"(buy_score={buy_score:.2f}, sell_score={sell_score:.2f}, "
            f"consensus={consensus_score:.2f})"
        )

        return action, consensus_score, strategy_signals

    def get_strategy_summary(self) -> Dict:
        """Get summary of all strategies"""
        return {
            'strategies': [
                {
                    'name': s.name,
                    'weight': s.weight,
                }
                for s in self.strategies
            ],
            'consensus_threshold': self.consensus_threshold,
            'total_weight': sum(s.weight for s in self.strategies),
        }

    def get_health_report(self) -> Dict:
        """
        Get health report for all strategies

        Returns:
            Dict containing health metrics for monitoring
        """
        # Calculate overall health metrics
        total_signals = sum(
            stats.get('total_signals', 0)
            for stats in self.strategy_health['strategy_stats'].values()
        )
        total_errors = sum(
            stats.get('error_count', 0)
            for stats in self.strategy_health['strategy_stats'].values()
        )

        # Log health report every 100 calls
        if self.strategy_health['total_calls'] % 100 == 0 and self.strategy_health['total_calls'] > 0:
            logger.info("=== Strategy Health Report ===")
            logger.info(f"Total analysis calls: {self.strategy_health['total_calls']}")

            for strategy_name, stats in self.strategy_health['strategy_stats'].items():
                signal_rate = (stats['buy_signals'] + stats['sell_signals']) / max(stats['total_signals'], 1)
                error_rate = stats['error_count'] / max(stats['total_signals'] + stats['error_count'], 1)

                logger.info(
                    f"{strategy_name}: "
                    f"Signals={stats['total_signals']} "
                    f"(Buy={stats['buy_signals']}, Sell={stats['sell_signals']}, Hold={stats['hold_signals']}), "
                    f"SignalRate={signal_rate:.1%}, "
                    f"AvgConf={stats['avg_confidence']:.2f}, "
                    f"Errors={stats['error_count']} ({error_rate:.1%})"
                )

            # Warn if any strategy has high error rate
            for strategy_name, stats in self.strategy_health['strategy_stats'].items():
                total_attempts = stats['total_signals'] + stats['error_count']
                if total_attempts > 10:  # Only check after reasonable sample size
                    error_rate = stats['error_count'] / total_attempts
                    if error_rate > 0.1:  # More than 10% errors
                        logger.warning(
                            f"HIGH ERROR RATE: {strategy_name} has {error_rate:.1%} error rate "
                            f"({stats['error_count']} errors in {total_attempts} attempts)"
                        )

                    # Check if strategy is generating signals
                    signal_rate = (stats['buy_signals'] + stats['sell_signals']) / stats['total_signals']
                    if signal_rate < 0.05 and stats['total_signals'] > 50:  # Less than 5% signals after 50+ attempts
                        logger.warning(
                            f"LOW SIGNAL RATE: {strategy_name} only generating {signal_rate:.1%} actionable signals "
                            f"(may need tuning or market conditions unsuitable)"
                        )

        return {
            'total_calls': self.strategy_health['total_calls'],
            'total_signals': total_signals,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_signals, 1),
            'strategy_stats': self.strategy_health['strategy_stats']
        }
