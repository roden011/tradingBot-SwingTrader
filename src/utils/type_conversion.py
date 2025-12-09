"""
Type Conversion Utilities

Provides safe type conversion functions with proper error handling
to prevent runtime crashes from invalid data.
"""

from typing import Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0, field_name: str = "value") -> float:
    """
    Safely convert a value to float with error handling.

    Args:
        value: Value to convert (can be str, int, float, None, etc.)
        default: Default value to return if conversion fails
        field_name: Name of the field being converted (for logging)

    Returns:
        Float value or default if conversion fails

    Examples:
        >>> safe_float("123.45")
        123.45
        >>> safe_float(None, default=0.0)
        0.0
        >>> safe_float("invalid", default=100.0)
        100.0
    """
    if value is None:
        logger.debug(f"{field_name} is None, returning default: {default}")
        return default

    if isinstance(value, float):
        return value

    if isinstance(value, (int, bool)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        if not value:
            logger.debug(f"{field_name} is empty string, returning default: {default}")
            return default

    try:
        return float(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert {field_name}={value} to float: {e}. Using default: {default}")
        return default


def safe_int(value: Any, default: int = 0, field_name: str = "value") -> int:
    """
    Safely convert a value to int with error handling.

    Args:
        value: Value to convert (can be str, int, float, None, etc.)
        default: Default value to return if conversion fails
        field_name: Name of the field being converted (for logging)

    Returns:
        Int value or default if conversion fails
    """
    if value is None:
        logger.debug(f"{field_name} is None, returning default: {default}")
        return default

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        value = value.strip()
        if not value:
            logger.debug(f"{field_name} is empty string, returning default: {default}")
            return default

    try:
        # Convert to float first to handle strings like "123.45"
        return int(float(value))
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert {field_name}={value} to int: {e}. Using default: {default}")
        return default


def safe_percentage(value: Any, default: float = 0.0, field_name: str = "value") -> float:
    """
    Safely convert a value to percentage (as decimal).

    Handles both decimal (0.05) and percentage (5.0) formats.

    Args:
        value: Value to convert
        default: Default value to return if conversion fails
        field_name: Name of the field being converted (for logging)

    Returns:
        Float value as decimal (e.g., 0.05 for 5%)
    """
    float_val = safe_float(value, default, field_name)

    # If value is > 1.0, assume it's in percentage format (e.g., 5.0 for 5%)
    # Convert to decimal (e.g., 0.05)
    if float_val > 1.0:
        logger.debug(f"{field_name}={float_val} appears to be percentage format, converting to decimal")
        return float_val / 100.0

    return float_val


def validate_positive(value: float, field_name: str = "value", allow_zero: bool = False) -> bool:
    """
    Validate that a numeric value is positive (or zero if allowed).

    Args:
        value: Value to validate
        field_name: Name of the field being validated (for logging)
        allow_zero: Whether to allow zero values

    Returns:
        True if valid, False otherwise
    """
    if allow_zero:
        if value < 0:
            logger.error(f"{field_name}={value} is negative (must be >= 0)")
            return False
    else:
        if value <= 0:
            logger.error(f"{field_name}={value} is not positive (must be > 0)")
            return False

    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0,
                field_name: str = "division") -> float:
    """
    Safely perform division with zero-check.

    Args:
        numerator: Top value
        denominator: Bottom value
        default: Default value to return if denominator is zero
        field_name: Name of the operation (for logging)

    Returns:
        Result of division or default if denominator is zero
    """
    if denominator == 0:
        logger.warning(f"{field_name}: Cannot divide {numerator} by zero, returning default: {default}")
        return default

    try:
        return numerator / denominator
    except (ZeroDivisionError, OverflowError) as e:
        logger.error(f"{field_name}: Division error ({numerator}/{denominator}): {e}. Using default: {default}")
        return default
