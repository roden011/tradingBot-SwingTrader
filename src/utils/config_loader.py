"""
Configuration Loader

Loads trading configuration from S3 with caching support.
Replaces the previous TRADING_CONFIG environment variable approach
to overcome the 4KB Lambda environment variable size limit.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
import boto3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Global cache for config (persists across Lambda warm starts)
_config_cache: Optional[Dict[str, Any]] = None
_cache_timestamp: Optional[datetime] = None
_cache_ttl_seconds = 300  # 5 minutes


class ConfigLoader:
    """
    Loads trading configuration from S3.

    Benefits over environment variables:
    - No 4KB size limit
    - Easier to update (upload new config without redeployment)
    - Supports versioning via S3
    - Can store full config with comments
    """

    def __init__(
        self,
        bucket_name: str,
        environment: str,
        profile: str,
        stage: str,
        cache_ttl_seconds: int = 300
    ):
        """
        Initialize config loader

        Args:
            bucket_name: S3 bucket name
            environment: Environment (dev, prod)
            profile: Trading profile (swing-trader, etc.)
            stage: Stage (blue, green)
            cache_ttl_seconds: Cache TTL in seconds (default: 300 = 5 minutes)
        """
        self.bucket_name = bucket_name
        self.environment = environment
        self.profile = profile
        self.stage = stage
        self.cache_ttl_seconds = cache_ttl_seconds
        self.s3_client = boto3.client('s3')

        # S3 key for config file
        # Format: config/{environment}-{profile}-{stage}.json
        self.config_key = f"config/{environment}-{profile}-{stage}.json"

        logger.info(
            f"ConfigLoader initialized: bucket={bucket_name}, "
            f"key={self.config_key}, cache_ttl={cache_ttl_seconds}s"
        )

    def load_config(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load configuration from S3 with optional caching

        Args:
            use_cache: Use cached config if available (default: True)

        Returns:
            Configuration dictionary

        Raises:
            Exception: If config cannot be loaded
        """
        global _config_cache, _cache_timestamp

        # Check cache
        if use_cache and _config_cache and _cache_timestamp:
            cache_age = (datetime.utcnow() - _cache_timestamp).total_seconds()
            if cache_age < self.cache_ttl_seconds:
                logger.debug(f"Using cached config (age: {cache_age:.1f}s)")
                return _config_cache

        try:
            logger.info(f"Loading config from S3: s3://{self.bucket_name}/{self.config_key}")

            # Download config from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.config_key
            )

            # Parse JSON
            config_str = response['Body'].read().decode('utf-8')
            config = json.loads(config_str)

            # Strip comment keys (those starting with '_')
            config = self._strip_comments(config)

            # Update cache
            _config_cache = config
            _cache_timestamp = datetime.utcnow()

            logger.info(f"Config loaded successfully from S3 (size: {len(config_str)} bytes)")

            return config

        except self.s3_client.exceptions.NoSuchKey:
            logger.error(f"Config file not found: s3://{self.bucket_name}/{self.config_key}")
            raise Exception(f"Config file not found in S3: {self.config_key}")

        except Exception as e:
            logger.error(f"Error loading config from S3: {e}")
            # If we have a cached config, use it as fallback
            if _config_cache:
                logger.warning("Falling back to cached config due to S3 error")
                return _config_cache
            raise

    def _strip_comments(self, obj):
        """Recursively remove keys starting with '_' from dict/list structures"""
        if isinstance(obj, dict):
            return {k: self._strip_comments(v) for k, v in obj.items() if not k.startswith('_')}
        elif isinstance(obj, list):
            return [self._strip_comments(item) for item in obj]
        else:
            return obj

    @staticmethod
    def from_env() -> 'ConfigLoader':
        """
        Create ConfigLoader from environment variables

        Returns:
            ConfigLoader instance

        Raises:
            KeyError: If required environment variables are missing
        """
        bucket_name = os.environ['DATA_BUCKET']
        environment = os.environ['ENVIRONMENT']
        profile = os.environ['PROFILE']
        stage = os.environ['STAGE']

        return ConfigLoader(bucket_name, environment, profile, stage)

    def invalidate_cache(self):
        """Invalidate the global config cache"""
        global _config_cache, _cache_timestamp
        _config_cache = None
        _cache_timestamp = None
        logger.info("Config cache invalidated")


def load_config_from_s3(
    bucket_name: str = None,
    environment: str = None,
    profile: str = None,
    stage: str = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to load config from S3

    Args:
        bucket_name: S3 bucket (defaults to DATA_BUCKET env var)
        environment: Environment (defaults to ENVIRONMENT env var)
        profile: Profile (defaults to PROFILE env var)
        stage: Stage (defaults to STAGE env var)
        use_cache: Use cached config if available (default: True)

    Returns:
        Configuration dictionary
    """
    bucket_name = bucket_name or os.environ['DATA_BUCKET']
    environment = environment or os.environ['ENVIRONMENT']
    profile = profile or os.environ['PROFILE']
    stage = stage or os.environ['STAGE']

    loader = ConfigLoader(bucket_name, environment, profile, stage)
    return loader.load_config(use_cache=use_cache)
