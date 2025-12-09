#!/usr/bin/env python3
"""Submit stop orders to Alpaca for ARES and PSLV positions."""

import json
import boto3
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Get credentials from Secrets Manager
secrets = boto3.client("secretsmanager", region_name="us-east-2")
secret_value = secrets.get_secret_value(SecretId="trading-bot/alpaca-credentials-dev-day-trader-blue")
creds = json.loads(secret_value["SecretString"])

api_key = creds["api_key"]
api_secret = creds["secret_key"]

# Create Alpaca client
client = TradingClient(api_key, api_secret, paper=True)

# Check existing orders
print("Checking existing stop orders...")
orders = client.get_orders()
existing_stops = {}
for order in orders:
    if order.type == "stop" and order.symbol in ["ARES", "PSLV"]:
        existing_stops[order.symbol] = order
        print(f"  Found: {order.symbol} stop @ ${order.stop_price} x {order.qty} ({order.status})")

if not existing_stops:
    print("  No existing stop orders for ARES or PSLV")

# Positions to protect
positions = [
    {"symbol": "ARES", "quantity": 325, "stop_price": 173.44},
    {"symbol": "PSLV", "quantity": 2984, "stop_price": 19.42}
]

print("\nSubmitting stop orders...")
for pos in positions:
    symbol = pos["symbol"]

    if symbol in existing_stops:
        print(f"  SKIP {symbol}: Stop already exists")
        continue

    try:
        stop_order = StopOrderRequest(
            symbol=symbol,
            qty=pos["quantity"],
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(pos["stop_price"], 2)
        )

        order = client.submit_order(order_data=stop_order)
        print(f"  OK {symbol}: Stop @ ${pos['stop_price']:.2f} for {pos['quantity']} shares (id: {order.id})")
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")

print("\nDone!")
