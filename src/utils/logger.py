"""
Logger utility for trading bot

Provides consistent logging configuration across all modules.
"""

import logging
import sys


def setup_logger(name: str, level: str = None) -> logging.Logger:
    """
    Setup a logger with consistent formatting

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, uses INFO

    Returns:
        Configured logger instance
    """
    # Get logger
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Set level
        log_level = getattr(logging, level.upper()) if level else logging.INFO
        logger.setLevel(log_level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add formatter to handler
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger
