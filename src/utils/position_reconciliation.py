"""
Position Reconciliation Utility

Ensures DynamoDB state matches Alpaca reality by comparing and syncing positions.
Critical for preventing state drift from partial fills, manual interventions, or corporate actions.
"""
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PositionDiscrepancy:
    """Types of position discrepancies"""
    QUANTITY_MISMATCH = "quantity_mismatch"
    MISSING_IN_DB = "missing_in_db"
    GHOST_POSITION = "ghost_position"


@dataclass
class ReconciliationResult:
    """Result of position reconciliation"""
    is_synced: bool
    discrepancies_found: int
    positions_added: List[str]
    positions_removed: List[str]
    quantities_adjusted: List[Tuple[str, float, float]]  # (symbol, old_qty, new_qty)
    details: str

    def __repr__(self):
        if self.is_synced:
            return f"ReconciliationResult(SYNCED, {self.discrepancies_found} discrepancies resolved)"
        else:
            return f"ReconciliationResult(OUT_OF_SYNC, {self.discrepancies_found} discrepancies)"


def reconcile_positions(
    alpaca_positions: Dict[str, Dict],
    local_positions: Dict[str, Dict],
    auto_sync: bool = True,
    tolerance: float = 0.01
) -> ReconciliationResult:
    """
    Compare Alpaca positions with local state and identify discrepancies

    Args:
        alpaca_positions: Positions from Alpaca API (source of truth)
        local_positions: Positions from local state (DynamoDB/memory)
        auto_sync: Whether to automatically sync local with Alpaca
        tolerance: Quantity difference tolerance (for floating point comparison)

    Returns:
        ReconciliationResult with sync status and details

    Note:
        Alpaca is ALWAYS the source of truth. Local state is updated to match.
    """
    positions_added = []
    positions_removed = []
    quantities_adjusted = []
    discrepancies = []

    # Get all symbols from both sources
    alpaca_symbols = set(alpaca_positions.keys())
    local_symbols = set(local_positions.keys())

    # Check for positions in Alpaca but not in local
    missing_from_local = alpaca_symbols - local_symbols
    if missing_from_local:
        discrepancies.append(
            f"Found {len(missing_from_local)} positions in Alpaca not in local state: {missing_from_local}"
        )
        positions_added.extend(missing_from_local)

    # Check for positions in local but not in Alpaca
    missing_from_alpaca = local_symbols - alpaca_symbols
    if missing_from_alpaca:
        discrepancies.append(
            f"Found {len(missing_from_alpaca)} positions in local state not in Alpaca: {missing_from_alpaca}"
        )
        positions_removed.extend(missing_from_alpaca)

    # Check for quantity mismatches in common positions
    common_symbols = alpaca_symbols & local_symbols
    for symbol in common_symbols:
        alpaca_qty = alpaca_positions[symbol].get('quantity', 0)
        local_qty = local_positions[symbol].get('quantity', 0)

        # Compare with tolerance for floating point
        if abs(alpaca_qty - local_qty) > tolerance:
            discrepancies.append(
                f"{symbol}: quantity mismatch - Alpaca has {alpaca_qty}, "
                f"local has {local_qty}"
            )
            quantities_adjusted.append((symbol, local_qty, alpaca_qty))

    # Prepare result
    total_discrepancies = len(discrepancies)

    if total_discrepancies == 0:
        logger.info("✓ Position reconciliation: all positions in sync")
        return ReconciliationResult(
            is_synced=True,
            discrepancies_found=0,
            positions_added=[],
            positions_removed=[],
            quantities_adjusted=[],
            details="All positions match between Alpaca and local state"
        )
    else:
        details = "\n".join(discrepancies)
        # Log at INFO level - reconciliation is routine state syncing, not an error
        logger.info(
            f"Position reconciliation: syncing {total_discrepancies} discrepancy(ies) "
            f"(normal after trades or manual interventions)"
        )
        logger.debug(f"Discrepancy details:\n{details}")

        if auto_sync:
            logger.info("Auto-syncing local state to match Alpaca (source of truth)")

        return ReconciliationResult(
            is_synced=False,
            discrepancies_found=total_discrepancies,
            positions_added=positions_added,
            positions_removed=positions_removed,
            quantities_adjusted=quantities_adjusted,
            details=details
        )


def sync_positions_to_alpaca(
    alpaca_positions: Dict[str, Dict],
    reconciliation_result: ReconciliationResult
) -> Dict[str, Dict]:
    """
    Update local positions to match Alpaca (source of truth)

    Args:
        alpaca_positions: Current positions from Alpaca
        reconciliation_result: Result from reconcile_positions

    Returns:
        Updated local positions dict matching Alpaca
    """
    if reconciliation_result.is_synced:
        logger.info("No sync needed, positions already match")
        return alpaca_positions.copy()

    logger.info(
        f"Syncing positions: "
        f"+{len(reconciliation_result.positions_added)} "
        f"-{len(reconciliation_result.positions_removed)} "
        f"~{len(reconciliation_result.quantities_adjusted)}"
    )

    # Return Alpaca positions as the new source of truth
    return alpaca_positions.copy()


class PositionReconciler:
    """
    Position Reconciliation Manager

    Compares DynamoDB positions with Alpaca positions and syncs them.
    Alpaca is always treated as the source of truth.
    """

    def __init__(self, alpaca_client, dynamodb_table):
        """
        Args:
            alpaca_client: AlpacaClient instance
            dynamodb_table: DynamoDB table resource for positions
        """
        self.alpaca_client = alpaca_client
        self.dynamodb_table = dynamodb_table

    def reconcile(self) -> List[Dict]:
        """
        Reconcile positions between Alpaca and DynamoDB

        Returns:
            List of discrepancy dictionaries
        """
        # Get Alpaca positions
        alpaca_positions_list = self.alpaca_client.get_positions()
        alpaca_positions = {}
        for pos in alpaca_positions_list:
            alpaca_positions[pos.symbol] = {
                'quantity': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
            }

        # Get DynamoDB positions
        db_response = self.dynamodb_table.scan()
        db_positions = {}
        for item in db_response.get('Items', []):
            symbol = item.get('symbol')
            if symbol:
                db_positions[symbol] = {
                    'quantity': float(item.get('quantity', 0)),
                    'avg_entry_price': float(item.get('avg_entry_price', 0)),
                }

        # Find discrepancies
        discrepancies = []
        alpaca_symbols = set(alpaca_positions.keys())
        db_symbols = set(db_positions.keys())

        # Check for positions in Alpaca but not in DB
        missing_in_db = alpaca_symbols - db_symbols
        for symbol in missing_in_db:
            discrepancies.append({
                'type': PositionDiscrepancy.MISSING_IN_DB,
                'symbol': symbol,
                'details': f"Alpaca has {alpaca_positions[symbol]['quantity']} shares, DB has none"
            })
            # Sync: add to DB
            self._update_db_position(symbol, alpaca_positions[symbol])
            logger.info(f"Synced missing position: added {symbol} to DB")

        # Check for positions in DB but not in Alpaca (ghost positions)
        ghost_positions = db_symbols - alpaca_symbols
        for symbol in ghost_positions:
            discrepancies.append({
                'type': PositionDiscrepancy.GHOST_POSITION,
                'symbol': symbol,
                'details': f"DB has {db_positions[symbol]['quantity']} shares, Alpaca has none"
            })
            # Sync: remove from DB
            self._remove_db_position(symbol)
            logger.info(f"Synced ghost position: removed {symbol} from DB")

        # Check for quantity mismatches
        common_symbols = alpaca_symbols & db_symbols
        for symbol in common_symbols:
            alpaca_qty = alpaca_positions[symbol]['quantity']
            db_qty = db_positions[symbol]['quantity']

            if abs(alpaca_qty - db_qty) > 0.01:  # Tolerance for floating point
                discrepancies.append({
                    'type': PositionDiscrepancy.QUANTITY_MISMATCH,
                    'symbol': symbol,
                    'details': f"DB: {db_qty}, Alpaca: {alpaca_qty}"
                })
                # Sync: update DB to match Alpaca
                self._update_db_position(symbol, alpaca_positions[symbol])
                logger.info(f"Synced quantity mismatch: updated {symbol} in DB")

        return discrepancies

    def _update_db_position(self, symbol: str, position_data: Dict):
        """
        Update position in DynamoDB with Alpaca data.

        Uses update_item to preserve trading-specific fields (stop_loss_price,
        peak_price, trailing_stop_active, entry_timestamp) that Alpaca doesn't provide.
        Only updates fields that Alpaca is the source of truth for.
        Also recalculates cost_basis to keep it consistent with quantity * avg_entry_price.
        """
        from decimal import Decimal

        try:
            # Recalculate cost_basis to keep it consistent
            quantity = position_data['quantity']
            avg_entry_price = position_data['avg_entry_price']
            cost_basis = quantity * avg_entry_price

            self.dynamodb_table.update_item(
                Key={'symbol': symbol},
                UpdateExpression='SET quantity = :q, avg_entry_price = :aep, current_price = :cp, market_value = :mv, unrealized_pl = :upl, cost_basis = :cb',
                ExpressionAttributeValues={
                    ':q': Decimal(str(quantity)),
                    ':aep': Decimal(str(avg_entry_price)),
                    ':cp': Decimal(str(position_data.get('current_price', 0))),
                    ':mv': Decimal(str(position_data.get('market_value', 0))),
                    ':upl': Decimal(str(position_data.get('unrealized_pl', 0))),
                    ':cb': Decimal(str(cost_basis)),
                }
            )
        except Exception as e:
            logger.error(f"Error updating position {symbol} in DB: {e}")

    def _remove_db_position(self, symbol: str):
        """Remove position from DynamoDB"""
        try:
            self.dynamodb_table.delete_item(Key={'symbol': symbol})
        except Exception as e:
            logger.error(f"Error removing position {symbol} from DB: {e}")
