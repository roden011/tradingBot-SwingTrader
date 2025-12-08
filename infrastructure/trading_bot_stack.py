"""
AWS CDK Stack for Serverless Trading Bot
"""
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    BundlingOptions,
    Tags,
    aws_lambda as lambda_,
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_events_targets as targets,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_secretsmanager as secretsmanager,
    aws_ssm as ssm,
    aws_iam as iam,
    aws_logs as logs,
)
from constructs import Construct
import os
import json
from typing import Dict, Any


class TradingBotStack(Stack):
    """
    Main CDK stack for the serverless trading bot
    Supports multi-environment, multi-profile blue/green deployments
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: Dict[str, Any],
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Store configuration
        self.config = config
        self.deploy_env = config['environment']  # dev or prod
        self.deploy_profile = config['profile']  # balanced, aggressive, conservative, etc.
        self.deploy_stage = config['stage']  # blue or green
        # Replace underscores with hyphens for AWS resource name compatibility (S3, CloudFormation, etc.)
        self.stack_prefix = f"trading-bot-{self.deploy_env}-{self.deploy_profile.replace('_', '-')}-{self.deploy_stage}"

        # Determine if EventBridge should be enabled (stage-specific scheduling)
        self.eventbridge_enabled = config.get('eventbridge_enabled', True)

        # Apply tags to all resources
        for key, value in config.get('tags', {}).items():
            Tags.of(self).add(key, value)

        # ========== Secrets Manager ==========
        # Reference existing Alpaca API credentials (must be created manually)
        # Dev: Separate credentials per profile-stage (dev-balanced-blue, dev-balanced-green)
        # Prod: Separate credentials per profile, shared across stages (prod-aggressive, prod-conservative)
        secret_name = config.get('alpaca_secret_name', f'trading-bot/alpaca-credentials-{self.deploy_env}-{self.deploy_profile}-{self.deploy_stage}')
        self.alpaca_secret = secretsmanager.Secret.from_secret_name_v2(
            self,
            "AlpacaCredentials",
            secret_name=secret_name
        )

        # ========== DynamoDB Tables ==========
        # Profile and stage-specific table names to ensure complete isolation
        # Positions table
        self.positions_table = dynamodb.Table(
            self,
            "PositionsTable",
            table_name=f"{self.stack_prefix}-positions",
            partition_key=dynamodb.Attribute(
                name="symbol", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
        )

        # Trades table
        self.trades_table = dynamodb.Table(
            self,
            "TradesTable",
            table_name=f"{self.stack_prefix}-trades",
            partition_key=dynamodb.Attribute(
                name="trade_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
        )

        # Add GSI for querying by symbol
        self.trades_table.add_global_secondary_index(
            index_name="symbol-timestamp-index",
            partition_key=dynamodb.Attribute(
                name="symbol", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
        )

        # Risk metrics table
        self.risk_metrics_table = dynamodb.Table(
            self,
            "RiskMetricsTable",
            table_name=f"{self.stack_prefix}-risk-metrics",
            partition_key=dynamodb.Attribute(
                name="metric_type", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
        )

        # System state table (for kill switch, circuit breaker, etc.)
        self.system_state_table = dynamodb.Table(
            self,
            "SystemStateTable",
            table_name=f"{self.stack_prefix}-system-state",
            partition_key=dynamodb.Attribute(
                name="state_key", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Day trades table (for PDT rule tracking)
        self.day_trades_table = dynamodb.Table(
            self,
            "DayTradesTable",
            table_name=f"{self.stack_prefix}-day-trades",
            partition_key=dynamodb.Attribute(
                name="date", type=dynamodb.AttributeType.STRING  # Format: YYYY-MM-DD
            ),
            sort_key=dynamodb.Attribute(
                name="symbol", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            time_to_live_attribute="ttl",  # Auto-delete old records (after 30 days)
        )

        # Realized losses table (for wash sale rule tracking)
        self.realized_losses_table = dynamodb.Table(
            self,
            "RealizedLossesTable",
            table_name=f"{self.stack_prefix}-realized-losses",
            partition_key=dynamodb.Attribute(
                name="symbol", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="loss_date", type=dynamodb.AttributeType.STRING  # ISO 8601 timestamp
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            time_to_live_attribute="ttl",  # Auto-delete old records (after 60 days)
        )

        # Universe cache table (for caching MarketScanner results)
        self.universe_cache_table = dynamodb.Table(
            self,
            "UniverseCacheTable",
            table_name=f"{self.stack_prefix}-universe-cache",
            partition_key=dynamodb.Attribute(
                name="cache_key", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            time_to_live_attribute="ttl",  # Auto-delete expired cache entries
        )

        # Tax obligations table (for tracking tax liability on realized gains/losses)
        self.tax_obligations_table = dynamodb.Table(
            self,
            "TaxObligationsTable",
            table_name=f"{self.stack_prefix}-tax-obligations",
            partition_key=dynamodb.Attribute(
                name="trade_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
        )

        # Add GSI for querying by sell_date (for quarterly/annual reporting)
        self.tax_obligations_table.add_global_secondary_index(
            index_name="sell-date-index",
            partition_key=dynamodb.Attribute(
                name="sell_date", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
        )

        # ========== Parameter Store ==========
        # Create SSM parameter for use_margin setting
        # This allows toggling margin trading without redeployment
        self.use_margin_parameter = ssm.StringParameter(
            self,
            "UseMarginParameter",
            parameter_name=f"/trading-bot/{self.deploy_env}/{self.deploy_profile}/{self.deploy_stage}/use_margin",
            string_value="true",  # Default value: margin enabled
            description=f"Controls margin trading for {self.deploy_env}-{self.deploy_profile}-{self.deploy_stage}. Set to 'true' to enable margin, 'false' to disable.",
            tier=ssm.ParameterTier.STANDARD,
        )

        # ========== S3 Buckets ==========
        # Bucket for blacklist and archives (profile and stage-specific)
        self.data_bucket = s3.Bucket(
            self,
            "DataBucket",
            bucket_name=f"{self.stack_prefix}-data-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ArchiveOldTrades",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        )
                    ],
                )
            ],
        )

        # Upload config file to S3
        # This deployment replaces the TRADING_CONFIG env var approach
        # to overcome the 4KB Lambda environment variable size limit
        config_file_name = f"{self.deploy_env}-{self.deploy_profile}-{self.deploy_stage}.json"
        self.config_deployment = s3deploy.BucketDeployment(
            self,
            "ConfigDeployment",
            sources=[s3deploy.Source.asset("config", exclude=["*", f"!{config_file_name}"])],
            destination_bucket=self.data_bucket,
            destination_key_prefix="config",
            prune=False,  # Don't delete other configs in S3
            retain_on_delete=True,  # Keep config in S3 if stack is deleted
        )

        # ========== SNS Topics ==========
        # Alert topic for notifications (profile and stage-specific)
        self.alert_topic = sns.Topic(
            self,
            "AlertTopic",
            topic_name=f"{self.stack_prefix}-alerts",
            display_name=f"Trading Bot Alerts - {self.deploy_env.upper()} - {self.deploy_profile.upper()} - {self.deploy_stage.upper()}",
        )

        # Add email subscription if provided
        alert_email = config.get("alert_email") or os.environ.get("ALERT_EMAIL")
        if alert_email:
            self.alert_topic.add_subscription(
                subscriptions.EmailSubscription(alert_email)
            )

        # ========== Lambda Layers ==========
        # Create a layer for dependencies
        self.dependencies_layer = lambda_.LayerVersion(
            self,
            "DependenciesLayer",
            code=lambda_.Code.from_asset("lambda_layer"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_11],
            description="Trading bot dependencies (pandas, numpy, alpaca-py)",
        )

        # ========== IAM Roles ==========
        # Lambda execution role with necessary permissions
        lambda_role = iam.Role(
            self,
            "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Grant permissions
        self.positions_table.grant_read_write_data(lambda_role)
        self.trades_table.grant_read_write_data(lambda_role)
        self.risk_metrics_table.grant_read_write_data(lambda_role)
        self.system_state_table.grant_read_write_data(lambda_role)
        self.day_trades_table.grant_read_write_data(lambda_role)
        self.realized_losses_table.grant_read_write_data(lambda_role)
        self.universe_cache_table.grant_read_write_data(lambda_role)
        self.tax_obligations_table.grant_read_write_data(lambda_role)
        self.data_bucket.grant_read_write(lambda_role)
        self.alpaca_secret.grant_read(lambda_role)
        self.alert_topic.grant_publish(lambda_role)
        self.use_margin_parameter.grant_read(lambda_role)

        # ========== Lambda Functions ==========
        # Helper function to remove comment keys (those starting with '_')
        def strip_comments(obj):
            """Recursively remove keys starting with '_' from dict/list structures"""
            if isinstance(obj, dict):
                return {k: strip_comments(v) for k, v in obj.items() if not k.startswith('_')}
            elif isinstance(obj, list):
                return [strip_comments(item) for item in obj]
            else:
                return obj

        # Common environment variables
        # NOTE: Trading config is now loaded from S3 instead of environment variables
        # to overcome the 4KB Lambda env var size limit
        common_env = {
            "POSITIONS_TABLE": self.positions_table.table_name,
            "TRADES_TABLE": self.trades_table.table_name,
            "RISK_METRICS_TABLE": self.risk_metrics_table.table_name,
            "SYSTEM_STATE_TABLE": self.system_state_table.table_name,
            "DAY_TRADES_TABLE": self.day_trades_table.table_name,
            "REALIZED_LOSSES_TABLE": self.realized_losses_table.table_name,
            "UNIVERSE_CACHE_TABLE": self.universe_cache_table.table_name,
            "TAX_OBLIGATIONS_TABLE": self.tax_obligations_table.table_name,
            "DATA_BUCKET": self.data_bucket.bucket_name,
            "ALERT_TOPIC_ARN": self.alert_topic.topic_arn,
            "ALPACA_SECRET_NAME": secret_name,
            "USE_MARGIN_PARAMETER_NAME": self.use_margin_parameter.parameter_name,
            "LOG_LEVEL": config.get("lambda", {}).get("log_level", "INFO"),
            # Deployment identification
            "ENVIRONMENT": self.deploy_env,
            "PROFILE": self.deploy_profile,  # Trading profile (swing-trader, etc.)
            "STAGE": self.deploy_stage,
            # Legacy env vars (kept for backward compatibility with old code)
            "DEPLOY_ENVIRONMENT": self.deploy_env,
            "DEPLOY_PROFILE": self.deploy_profile,
            "DEPLOY_STAGE": self.deploy_stage,
        }

        # Trading Executor Lambda (profile and stage-specific function name)
        self.trading_executor = lambda_.Function(
            self,
            "TradingExecutor",
            function_name=f"{self.stack_prefix}-executor",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambdas.trading_executor.handler.lambda_handler",
            code=lambda_.Code.from_asset(
                "src",
                exclude=["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/test_*", "**/.pytest_cache"]
            ),
            role=lambda_role,
            timeout=Duration.minutes(config.get("lambda", {}).get("timeout_minutes", 10)),
            memory_size=config.get("lambda", {}).get("memory_size", 1024),
            layers=[self.dependencies_layer],
            environment=common_env,
        )

        # Kill Switch Lambda (profile and stage-specific function name)
        self.kill_switch = lambda_.Function(
            self,
            "KillSwitch",
            function_name=f"{self.stack_prefix}-kill-switch",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambdas.kill_switch.handler.lambda_handler",
            code=lambda_.Code.from_asset(
                "src",
                exclude=["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/test_*", "**/.pytest_cache"]
            ),
            role=lambda_role,
            timeout=Duration.minutes(2),
            memory_size=256,
            layers=[self.dependencies_layer],
            environment=common_env,
        )

        # Daily Report Lambda (profile and stage-specific function name)
        self.daily_report = lambda_.Function(
            self,
            "DailyReport",
            function_name=f"{self.stack_prefix}-daily-report",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambdas.daily_report.handler.lambda_handler",
            code=lambda_.Code.from_asset(
                "src",
                exclude=["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/test_*", "**/.pytest_cache"]
            ),
            role=lambda_role,
            timeout=Duration.minutes(2),
            memory_size=256,
            layers=[self.dependencies_layer],
            environment=common_env,
        )

        # Quarterly Tax Report Lambda (profile and stage-specific function name)
        self.quarterly_report = lambda_.Function(
            self,
            "QuarterlyReport",
            function_name=f"{self.stack_prefix}-quarterly-report",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambdas.quarterly_report.handler.lambda_handler",
            code=lambda_.Code.from_asset(
                "src",
                exclude=["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/test_*", "**/.pytest_cache"]
            ),
            role=lambda_role,
            timeout=Duration.minutes(2),
            memory_size=256,
            layers=[self.dependencies_layer],
            environment=common_env,
        )

        # Weekly Report Lambda (profile and stage-specific function name)
        self.weekly_report = lambda_.Function(
            self,
            "WeeklyReport",
            function_name=f"{self.stack_prefix}-weekly-report",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambdas.weekly_report.handler.lambda_handler",
            code=lambda_.Code.from_asset(
                "src",
                exclude=["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/test_*", "**/.pytest_cache"]
            ),
            role=lambda_role,
            timeout=Duration.minutes(2),
            memory_size=256,
            layers=[self.dependencies_layer],
            environment=common_env,
        )

        # Monthly Report Lambda (profile and stage-specific function name)
        self.monthly_report = lambda_.Function(
            self,
            "MonthlyReport",
            function_name=f"{self.stack_prefix}-monthly-report",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambdas.monthly_report.handler.lambda_handler",
            code=lambda_.Code.from_asset(
                "src",
                exclude=["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/test_*", "**/.pytest_cache"]
            ),
            role=lambda_role,
            timeout=Duration.minutes(3),
            memory_size=512,
            layers=[self.dependencies_layer],
            environment=common_env,
        )

        # Test NumPy Lambda (temporary for debugging)
        self.test_numpy = lambda_.Function(
            self,
            "TestNumPy",
            function_name=f"{self.stack_prefix}-test-numpy",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="test_numpy_lambda.lambda_handler",
            code=lambda_.Code.from_asset(
                "src",
                exclude=["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/.pytest_cache"]
            ),
            role=lambda_role,
            timeout=Duration.seconds(30),
            memory_size=256,
            layers=[self.dependencies_layer],
            environment=common_env,
        )

        # ========== EventBridge Rules ==========
        # Conditionally enable EventBridge based on configuration
        # Dev: Both stages active (eventbridge_enabled=true for both)
        # Prod: Only active stage has eventbridge_enabled=true

        # Get schedule configuration
        schedule_config = config.get("schedule", {})
        trading_interval = schedule_config.get("trading_interval_minutes", 15)
        trading_start_hour_et = schedule_config.get("trading_start_hour_et", 9)
        trading_start_minute_et = schedule_config.get("trading_start_minute_et", 30)
        trading_end_hour_et = schedule_config.get("trading_end_hour_et", 16)
        trading_end_minute_et = schedule_config.get("trading_end_minute_et", 0)
        daily_report_hour_et = schedule_config.get("daily_report_hour_et", 16)
        daily_report_minute_et = schedule_config.get("daily_report_minute_et", 30)

        # Convert ET to UTC (ET = UTC-5, standard time offset)
        # Note: EventBridge uses UTC. For simplicity, we use standard time offset (EST = UTC-5)
        # TODO: Consider daylight saving time if needed (EDT = UTC-4)
        trading_start_hour_utc = trading_start_hour_et + 5
        trading_end_hour_utc = trading_end_hour_et + 5
        daily_report_hour_utc = daily_report_hour_et + 5

        # Build minute list for trading execution based on interval
        # Generate minutes: 0, interval, interval*2, ... up to 59
        trading_minutes = [str(m) for m in range(0, 60, trading_interval)]
        trading_minutes_str = ",".join(trading_minutes)

        # Build hour range for trading (handle edge cases)
        # If start time has minutes, we need to include that hour
        # If end time is exactly on the hour, we stop at the previous hour
        if trading_start_minute_et > 0:
            # Start from the hour where trading begins
            start_hour_utc = trading_start_hour_utc
        else:
            start_hour_utc = trading_start_hour_utc

        if trading_end_minute_et == 0:
            # End at the previous hour if end time is exactly on the hour
            end_hour_utc = trading_end_hour_utc - 1
        else:
            end_hour_utc = trading_end_hour_utc

        # Handle day wrap-around (e.g., if UTC hours cross midnight)
        if end_hour_utc >= 24:
            end_hour_utc = end_hour_utc - 24

        trading_hours_str = f"{start_hour_utc}-{end_hour_utc}"

        if self.eventbridge_enabled:
            # Trading execution rule (configurable interval during market hours)
            # Note: Market hours are configurable via schedule parameters
            # Default: Mon-Fri, 9:30 AM - 4:00 PM ET
            trading_rule = events.Rule(
                self,
                "TradingExecutionRule",
                schedule=events.Schedule.cron(
                    minute=trading_minutes_str,
                    hour=trading_hours_str,
                    week_day="MON-FRI",
                ),
                description=f"Execute trading strategies every {trading_interval}min - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()}",
                enabled=True,
            )
            trading_rule.add_target(targets.LambdaFunction(self.trading_executor))

            # Daily report rule (configurable time)
            daily_report_rule = events.Rule(
                self,
                "DailyReportRule",
                schedule=events.Schedule.cron(
                    minute=str(daily_report_minute_et),
                    hour=str(daily_report_hour_utc),
                    week_day="MON-FRI",
                ),
                description=f"Generate daily trading report - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()}",
                enabled=True,
            )
            daily_report_rule.add_target(targets.LambdaFunction(self.daily_report))

            # Weekly report rule (Friday 5:00 PM ET)
            # 5:00 PM EST = 22:00 UTC, 5:00 PM EDT = 21:00 UTC
            # We use 22:00 UTC to ensure after market close during EST (Nov-Mar)
            weekly_report_rule = events.Rule(
                self,
                "WeeklyReportRule",
                schedule=events.Schedule.cron(
                    minute="0",
                    hour="22",  # 5:00 PM EST / 6:00 PM EDT
                    week_day="FRI",
                ),
                description=f"Generate weekly trading report - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()}",
                enabled=True,
            )
            weekly_report_rule.add_target(targets.LambdaFunction(self.weekly_report))

            # Monthly report rule (Last day of month at 5:00 PM ET)
            # Using day=L (last day of month)
            # 5:00 PM EST = 22:00 UTC, 5:00 PM EDT = 21:00 UTC
            monthly_report_rule = events.Rule(
                self,
                "MonthlyReportRule",
                schedule=events.Schedule.cron(
                    minute="0",
                    hour="22",  # 5:00 PM EST / 6:00 PM EDT
                    day="L",  # Last day of the month
                ),
                description=f"Generate monthly trading report - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()}",
                enabled=True,
            )
            monthly_report_rule.add_target(targets.LambdaFunction(self.monthly_report))

            # Quarterly tax report rules (last day of each quarter at 5:00 PM ET)
            # Q1: March 31, Q2: June 30, Q3: September 30, Q4: December 31
            for quarter, month in [(1, 3), (2, 6), (3, 9), (4, 12)]:
                quarterly_report_rule = events.Rule(
                    self,
                    f"QuarterlyReportQ{quarter}Rule",
                    schedule=events.Schedule.cron(
                        minute="0",
                        hour="22",  # 5:00 PM EST / 6:00 PM EDT
                        day="L",  # Last day of the month
                        month=str(month),
                    ),
                    description=f"Generate Q{quarter} tax report - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()}",
                    enabled=True,
                )
                quarterly_report_rule.add_target(targets.LambdaFunction(self.quarterly_report))
        else:
            # Create rules but keep them disabled for standby stages
            trading_rule = events.Rule(
                self,
                "TradingExecutionRule",
                schedule=events.Schedule.cron(
                    minute=trading_minutes_str,
                    hour=trading_hours_str,
                    week_day="MON-FRI",
                ),
                description=f"Execute trading strategies every {trading_interval}min - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()} (STANDBY)",
                enabled=False,
            )
            trading_rule.add_target(targets.LambdaFunction(self.trading_executor))

            daily_report_rule = events.Rule(
                self,
                "DailyReportRule",
                schedule=events.Schedule.cron(
                    minute=str(daily_report_minute_et),
                    hour=str(daily_report_hour_utc),
                    week_day="MON-FRI",
                ),
                description=f"Generate daily trading report - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()} (STANDBY)",
                enabled=False,
            )
            daily_report_rule.add_target(targets.LambdaFunction(self.daily_report))

            # Weekly report rule - disabled for standby
            weekly_report_rule = events.Rule(
                self,
                "WeeklyReportRule",
                schedule=events.Schedule.cron(
                    minute="0",
                    hour="22",  # 5:00 PM EST / 6:00 PM EDT
                    week_day="FRI",
                ),
                description=f"Generate weekly trading report - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()} (STANDBY)",
                enabled=False,
            )
            weekly_report_rule.add_target(targets.LambdaFunction(self.weekly_report))

            # Monthly report rule - disabled for standby
            monthly_report_rule = events.Rule(
                self,
                "MonthlyReportRule",
                schedule=events.Schedule.cron(
                    minute="0",
                    hour="22",  # 5:00 PM EST / 6:00 PM EDT
                    day="L",  # Last day of the month
                ),
                description=f"Generate monthly trading report - {self.deploy_env.upper()} {self.deploy_profile.upper()} {self.deploy_stage.upper()} (STANDBY)",
                enabled=False,
            )
            monthly_report_rule.add_target(targets.LambdaFunction(self.monthly_report))
