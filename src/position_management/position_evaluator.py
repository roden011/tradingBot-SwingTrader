"""
Position Evaluator - Evaluates position strength and manages intelligent rebalancing
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionScore:
    """Score for a position's strength"""
    symbol: str
    score: float  # 0.0 to 1.0 (higher = stronger)
    unrealized_pl_pct: float
    days_held: int
    momentum: float
    volatility: float
    current_price: float
    quantity: float
    market_value: float
    metadata: Dict


@dataclass
class RebalanceDecision:
    """Decision on whether to rebalance positions"""
    should_rebalance: bool
    positions_to_sell: List[str]  # Symbols to sell
    estimated_cash_freed: float
    reason: str
    new_buy_score: float
    weakest_position_score: float
    metadata: Dict


class PositionEvaluator:
    """
    Evaluates position strength and manages intelligent rebalancing decisions

    Position strength is evaluated based on:
    1. Current unrealized P&L percentage
    2. Recent momentum (price trend)
    3. Position volatility
    4. Time held (recency bias)
    5. Consensus score from strategies
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize position evaluator

        Args:
            config: Position management configuration dict
        """
        config = config or {}
        weights = config.get('scoring_weights', {})

        # Weights for position scoring (must sum to 1.0)
        self.pl_weight = weights.get('pl_weight', 0.35)
        self.momentum_weight = weights.get('momentum_weight', 0.30)
        self.volatility_weight = weights.get('volatility_weight', 0.15)
        self.time_weight = weights.get('time_weight', 0.10)
        self.consensus_weight = weights.get('consensus_weight', 0.10)

        # Additional config
        self.rebalancing_enabled = config.get('rebalancing_enabled', True)
        self.momentum_lookback_days = config.get('momentum_lookback_days', 5)
        self.volatility_lookback_days = config.get('volatility_lookback_days', 20)

        # Verify weights sum to 1.0
        total_weight = (self.pl_weight + self.momentum_weight + self.volatility_weight +
                       self.time_weight + self.consensus_weight)
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Position scoring weights don't sum to 1.0 (got {total_weight:.3f})")

        logger.info(
            f"Position evaluator initialized: "
            f"pl={self.pl_weight:.2f}, momentum={self.momentum_weight:.2f}, "
            f"volatility={self.volatility_weight:.2f}, rebalancing={self.rebalancing_enabled}"
        )

    def evaluate_position_strength(
        self,
        symbol: str,
        position_data: Dict,
        market_data: pd.DataFrame,
        consensus_score: Optional[float] = None
    ) -> PositionScore:
        """
        Evaluate the strength of a position

        Args:
            symbol: Stock symbol
            position_data: Position info (from Alpaca)
            market_data: Historical price data
            consensus_score: Original consensus score when position was opened

        Returns:
            PositionScore with overall strength rating
        """
        # Calculate individual components
        pl_score = self._calculate_pl_score(position_data)
        momentum_score = self._calculate_momentum_score(market_data)
        volatility_score = self._calculate_volatility_score(market_data)
        time_score = self._calculate_time_score(position_data)
        consensus_score = consensus_score or 0.5  # Default if not available

        # Calculate weighted overall score
        overall_score = (
            pl_score * self.pl_weight +
            momentum_score * self.momentum_weight +
            volatility_score * self.volatility_weight +
            time_score * self.time_weight +
            consensus_score * self.consensus_weight
        )

        # Calculate days held (simplified - would use actual entry timestamp in production)
        days_held = 1  # Placeholder

        return PositionScore(
            symbol=symbol,
            score=overall_score,
            unrealized_pl_pct=position_data.get('unrealized_pl', 0) / position_data.get('market_value', 1) * 100,
            days_held=days_held,
            momentum=momentum_score,
            volatility=volatility_score,
            current_price=position_data.get('current_price', 0),
            quantity=position_data.get('quantity', 0),
            market_value=position_data.get('market_value', 0),
            metadata={
                'pl_score': pl_score,
                'momentum_score': momentum_score,
                'volatility_score': volatility_score,
                'time_score': time_score,
                'consensus_score': consensus_score
            }
        )

    def _calculate_pl_score(self, position_data: Dict) -> float:
        """
        Calculate score based on unrealized P&L
        Positive P&L = higher score, negative P&L = lower score
        """
        unrealized_pl = position_data.get('unrealized_pl', 0)
        market_value = position_data.get('market_value', 1)

        pl_pct = (unrealized_pl / market_value) * 100 if market_value > 0 else 0

        # Normalize to 0-1 scale
        # -20% = 0.0, 0% = 0.5, +20% = 1.0
        score = 0.5 + (pl_pct / 40.0)

        # Clamp to 0-1
        return max(0.0, min(1.0, score))

    def _calculate_momentum_score(self, market_data: pd.DataFrame) -> float:
        """
        Calculate score based on recent price momentum
        Uses 5-day and 20-day price change
        """
        if market_data.empty or len(market_data) < 20:
            return 0.5  # Neutral if insufficient data

        try:
            current_price = market_data['close'].iloc[-1]
            price_5d_ago = market_data['close'].iloc[-5]
            price_20d_ago = market_data['close'].iloc[-20]

            # Calculate momentum
            momentum_5d = (current_price - price_5d_ago) / price_5d_ago
            momentum_20d = (current_price - price_20d_ago) / price_20d_ago

            # Weight recent momentum more heavily
            combined_momentum = (momentum_5d * 0.7) + (momentum_20d * 0.3)

            # Normalize to 0-1 scale
            # -10% = 0.0, 0% = 0.5, +10% = 1.0
            score = 0.5 + (combined_momentum / 0.2)

            # Clamp to 0-1
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Error calculating momentum score: {e}")
            return 0.5

    def _calculate_volatility_score(self, market_data: pd.DataFrame) -> float:
        """
        Calculate score based on volatility
        Lower volatility = higher score (more stable)
        """
        if market_data.empty or len(market_data) < 20:
            return 0.5

        try:
            # Calculate 20-day historical volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.tail(20).std() * (252 ** 0.5)  # Annualized

            # Normalize to 0-1 scale
            # 100% vol = 0.0, 20% vol = 0.5, 0% vol = 1.0
            score = 1.0 - (volatility / 2.0)

            # Clamp to 0-1
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Error calculating volatility score: {e}")
            return 0.5

    def _calculate_time_score(self, position_data: Dict) -> float:
        """
        Calculate score based on time held
        Slightly prefer newer positions (recency bias)
        """
        # Placeholder - would use actual entry timestamp in production
        # For now, return neutral score
        return 0.5

    def evaluate_rebalance_decision(
        self,
        candidate_symbol: str,
        candidate_consensus: float,
        current_positions: Dict,
        historical_data: Dict[str, pd.DataFrame],
        cash_needed: float
    ) -> tuple:
        """
        Evaluate whether to rebalance positions to free up cash for a new buy.

        IMPORTANT: Only sells positions that are WEAKER than the new buy candidate.
        Uses the same multi-factor scoring for both existing positions and new candidates
        to ensure an apples-to-apples comparison.

        Args:
            candidate_symbol: Symbol of potential new buy
            candidate_consensus: Consensus score for new buy (0.0-1.0)
            current_positions: Current positions dict
            historical_data: Historical market data for all symbols
            cash_needed: Amount of cash needed for new buy

        Returns:
            Tuple of (should_rebalance: bool, positions_to_sell: List[str])
        """
        logger.info(f"📊 Evaluating rebalance: {candidate_symbol} (consensus: {candidate_consensus:.3f}), need ${cash_needed:,.2f}")

        # Score all current positions using multi-factor evaluation
        position_scores = []
        for symbol, position in current_positions.items():
            if symbol in historical_data:
                score = self.evaluate_position_strength(
                    symbol,
                    position,
                    historical_data[symbol],
                    consensus_score=position.get('consensus_score')  # Use stored consensus if available
                )
                position_scores.append(score)
                logger.debug(f"  Position {symbol}: score={score.score:.3f}")
            else:
                logger.warning(f"No historical data for {symbol}, skipping evaluation")

        if not position_scores:
            logger.warning("No positions to evaluate for rebalancing")
            return False, []

        # Sort positions by score (weakest first)
        position_scores.sort(key=lambda x: x.score)

        # Score the NEW BUY candidate using the same multi-factor approach
        # This ensures apples-to-apples comparison
        new_buy_score = self._calculate_candidate_score(
            candidate_symbol,
            candidate_consensus,
            historical_data.get(candidate_symbol)
        )

        logger.info(f"  New buy {candidate_symbol} score: {new_buy_score:.3f}")
        logger.info(f"  Weakest position {position_scores[0].symbol} score: {position_scores[0].score:.3f}")

        # Find positions to sell - ONLY if weaker than new buy
        positions_to_sell = []
        cash_freed = 0.0

        for pos_score in position_scores:
            # CRITICAL CHECK: Only sell if position is weaker than new buy
            if pos_score.score >= new_buy_score:
                logger.info(
                    f"  ⛔ {pos_score.symbol} (score: {pos_score.score:.3f}) is STRONGER than "
                    f"{candidate_symbol} (score: {new_buy_score:.3f}) - keeping position"
                )
                break

            positions_to_sell.append(pos_score.symbol)
            cash_freed += pos_score.market_value

            logger.info(
                f"  ✅ {pos_score.symbol} (score: {pos_score.score:.3f}) is WEAKER than "
                f"{candidate_symbol} (score: {new_buy_score:.3f}) - candidate for sale "
                f"(value: ${pos_score.market_value:,.2f})"
            )

            # Stop if we've freed enough cash
            if cash_freed >= cash_needed:
                break

        # Determine if rebalancing is worthwhile
        should_rebalance = (
            len(positions_to_sell) > 0 and
            cash_freed >= cash_needed
        )

        if should_rebalance:
            logger.info(
                f"  🔄 REBALANCE APPROVED: Sell {positions_to_sell} to buy {candidate_symbol} "
                f"(freeing ${cash_freed:,.2f} of ${cash_needed:,.2f} needed)"
            )
        elif len(positions_to_sell) == 0:
            logger.info(
                f"  ❌ REBALANCE DENIED: All positions are stronger than {candidate_symbol}"
            )
        else:
            logger.info(
                f"  ❌ REBALANCE DENIED: Can only free ${cash_freed:,.2f} of ${cash_needed:,.2f} needed"
            )

        return should_rebalance, positions_to_sell

    def _calculate_candidate_score(
        self,
        symbol: str,
        consensus_score: float,
        market_data: pd.DataFrame
    ) -> float:
        """
        Calculate a comparable score for a new buy candidate.

        Uses the same factors as position scoring but with appropriate defaults
        for a position we don't yet hold:
        - P&L score: 0.5 (neutral - no P&L yet)
        - Momentum score: calculated from market data
        - Volatility score: calculated from market data
        - Time score: 1.0 (fresh entry is optimal)
        - Consensus score: the actual consensus from strategies

        Returns:
            Float score in same range as position scores (0.0-1.0)
        """
        # P&L score: neutral for new position (no gains or losses yet)
        pl_score = 0.5

        # Calculate momentum and volatility from market data if available
        if market_data is not None and not market_data.empty:
            momentum_score = self._calculate_momentum_score(market_data)
            volatility_score = self._calculate_volatility_score(market_data)
        else:
            momentum_score = 0.5  # Neutral if no data
            volatility_score = 0.5

        # Time score: 1.0 for new entries (best time to enter is now if signal is valid)
        time_score = 1.0

        # Consensus score: use the actual consensus (normalize if needed)
        # Consensus is typically 0.0-0.5, so we scale it to 0.0-1.0 for fair comparison
        normalized_consensus = min(1.0, consensus_score * 2.5)  # 0.4 consensus -> 1.0 score

        # Calculate weighted score using same weights as position scoring
        candidate_score = (
            pl_score * self.pl_weight +
            momentum_score * self.momentum_weight +
            volatility_score * self.volatility_weight +
            time_score * self.time_weight +
            normalized_consensus * self.consensus_weight
        )

        logger.debug(
            f"Candidate {symbol} score breakdown: "
            f"pl={pl_score:.2f}*{self.pl_weight}, "
            f"momentum={momentum_score:.2f}*{self.momentum_weight}, "
            f"volatility={volatility_score:.2f}*{self.volatility_weight}, "
            f"time={time_score:.2f}*{self.time_weight}, "
            f"consensus={normalized_consensus:.2f}*{self.consensus_weight} "
            f"= {candidate_score:.3f}"
        )

        return candidate_score

    def _generate_rebalance_reason(
        self,
        should_rebalance: bool,
        new_buy_symbol: str,
        new_buy_score: float,
        positions_to_sell: List[str],
        position_scores: List[PositionScore],
        cash_freed: float,
        cash_needed: float
    ) -> str:
        """Generate human-readable reason for rebalance decision"""
        if should_rebalance:
            weakest_symbols = ', '.join(positions_to_sell)
            weakest_scores = ', '.join([
                f"{ps.symbol}={ps.score:.3f}"
                for ps in position_scores[:len(positions_to_sell)]
            ])
            return (
                f"Selling {len(positions_to_sell)} weak position(s) [{weakest_symbols}] "
                f"(scores: {weakest_scores}) to buy stronger position {new_buy_symbol} "
                f"(score: {new_buy_score:.3f}). Cash freed: ${cash_freed:,.2f} (needed: ${cash_needed:,.2f})"
            )
        else:
            if not positions_to_sell:
                return (
                    f"All current positions are stronger than {new_buy_symbol} "
                    f"(new buy score: {new_buy_score:.3f}, weakest position score: "
                    f"{position_scores[0].score:.3f}). Not rebalancing."
                )
            else:
                return (
                    f"Cannot free enough cash (need ${cash_needed:,.2f}, can free ${cash_freed:,.2f}) "
                    f"without selling positions stronger than {new_buy_symbol}"
                )
