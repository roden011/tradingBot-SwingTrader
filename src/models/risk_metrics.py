"""
Risk Metrics data model for DynamoDB
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from tradingbot_core.models.utils import convert_floats_to_decimal


@dataclass
class RiskMetrics:
    """Represents risk metrics at a point in time"""

    metric_type: str  # 'daily', 'weekly', 'portfolio', 'position'
    timestamp: str
    portfolio_value: float
    cash_balance: float
    total_exposure: float
    exposure_pct: float
    daily_pl: float
    daily_pl_pct: float
    weekly_pl: float
    weekly_pl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    total_risk_pct: float  # Sum of all position risks
    sector_concentrations: dict
    position_count: int
    active_strategies: list
    win_rate: float
    consecutive_losses: int
    vix_level: float
    circuit_breaker_active: bool
    risk_violations: list

    @classmethod
    def create_snapshot(
        cls,
        metric_type: str,
        portfolio_value: float,
        cash_balance: float,
        total_exposure: float,
        daily_pl: float,
        weekly_pl: float,
        max_drawdown: float,
        sector_concentrations: dict,
        position_count: int,
        active_strategies: list,
        win_rate: float,
        consecutive_losses: int,
        vix_level: float,
        circuit_breaker_active: bool,
        risk_violations: list,
    ):
        """Create a risk metrics snapshot"""
        exposure_pct = (
            (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        )
        daily_pl_pct = (daily_pl / portfolio_value * 100) if portfolio_value > 0 else 0
        weekly_pl_pct = (
            (weekly_pl / portfolio_value * 100) if portfolio_value > 0 else 0
        )
        max_drawdown_pct = (
            (max_drawdown / portfolio_value * 100) if portfolio_value > 0 else 0
        )

        # Calculate total risk percentage
        total_risk_pct = exposure_pct * 0.02  # Assuming 2% risk per position

        return cls(
            metric_type=metric_type,
            timestamp=datetime.utcnow().isoformat(),
            portfolio_value=portfolio_value,
            cash_balance=cash_balance,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            daily_pl=daily_pl,
            daily_pl_pct=daily_pl_pct,
            weekly_pl=weekly_pl,
            weekly_pl_pct=weekly_pl_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_risk_pct=total_risk_pct,
            sector_concentrations=sector_concentrations,
            position_count=position_count,
            active_strategies=active_strategies,
            win_rate=win_rate,
            consecutive_losses=consecutive_losses,
            vix_level=vix_level,
            circuit_breaker_active=circuit_breaker_active,
            risk_violations=risk_violations,
        )

    def to_dynamodb_item(self) -> dict:
        """Convert to DynamoDB item format"""
        item = asdict(self)
        # Convert all floats to Decimal for DynamoDB
        return convert_floats_to_decimal(item)

    @classmethod
    def from_dynamodb_item(cls, item: dict):
        """Create RiskMetrics from DynamoDB item"""
        return cls(**item)
