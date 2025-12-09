"""
System State data model for DynamoDB
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from tradingbot_core.models.utils import convert_floats_to_decimal


@dataclass
class SystemState:
    """Represents system-wide state (kill switch, circuit breaker, etc.)"""

    state_key: str  # 'kill_switch', 'circuit_breaker', 'trading_enabled'
    value: bool
    last_updated: str
    updated_by: str  # 'system', 'user', 'risk_manager'
    reason: str
    metadata: dict

    @classmethod
    def create(
        cls, state_key: str, value: bool, updated_by: str, reason: str, metadata=None
    ):
        """Create a new system state"""
        return cls(
            state_key=state_key,
            value=value,
            last_updated=datetime.utcnow().isoformat(),
            updated_by=updated_by,
            reason=reason,
            metadata=metadata or {},
        )

    def to_dynamodb_item(self) -> dict:
        """Convert to DynamoDB item format"""
        return convert_floats_to_decimal(asdict(self))

    @classmethod
    def from_dynamodb_item(cls, item: dict):
        """Create SystemState from DynamoDB item"""
        return cls(**item)

    def update(self, value: bool, updated_by: str, reason: str):
        """Update the state"""
        self.value = value
        self.last_updated = datetime.utcnow().isoformat()
        self.updated_by = updated_by
        self.reason = reason
