#!/usr/bin/env python3
"""
CDK App entry point for serverless trading bot
Supports multi-profile blue/green deployment with environment-specific configurations
"""
import os
import json
import subprocess
from pathlib import Path
from aws_cdk import App, Environment
from infrastructure.trading_bot_stack import TradingBotStack


def get_git_commit_info():
    """Get git commit hash and short hash for tagging resources."""
    try:
        # Get full commit hash
        full_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        ).stdout.strip()

        # Get short hash
        short_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        ).stdout.strip()

        return full_hash, short_hash
    except Exception:
        return "unknown", "unknown"

app = App()

# Get deployment configuration from context
config_env = app.node.try_get_context("environment") or os.environ.get("DEPLOY_ENV", "dev")
config_profile = app.node.try_get_context("profile") or os.environ.get("DEPLOY_PROFILE", "swing-trader")
config_stage = app.node.try_get_context("stage") or os.environ.get("DEPLOY_STAGE", "blue")

# Load configuration file
config_path = Path(__file__).parent / "config" / f"{config_env}-{config_profile}-{config_stage}.json"
if not config_path.exists():
    raise ValueError(f"Configuration file not found: {config_path}")

with open(config_path, 'r') as f:
    config = json.load(f)

# Add git commit info to tags for deployment tracking
commit_full, commit_short = get_git_commit_info()
if 'tags' not in config:
    config['tags'] = {}
config['tags']['GitCommit'] = commit_short
config['tags']['GitCommitFull'] = commit_full

# Create CDK environment
env = Environment(
    account=config.get("account") or os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=config.get("region", "us-east-2")
)

# Create stack with unique name per environment, profile, and stage
# Replace underscores with hyphens for CloudFormation compatibility
stack_name = f"TradingBot-{config['environment']}-{config['profile'].replace('_', '-')}-{config['stage']}"

TradingBotStack(
    app,
    stack_name,
    config=config,
    env=env,
    description=f"Serverless Trading Bot - {config['environment'].upper()} - {config['profile'].upper()} - {config['stage'].upper()}"
)

app.synth()
