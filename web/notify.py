"""Trade notifications via Slack webhook."""

import json
import os
import urllib.request
from datetime import datetime, timezone


SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def send_slack(text, blocks=None):
    """Send a message to Slack via webhook."""
    if not SLACK_WEBHOOK_URL:
        print("[notify] SLACK_WEBHOOK_URL not set, skipping")
        return False

    payload = {"text": text}
    if blocks:
        payload["blocks"] = blocks

    try:
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[notify] Slack error: {e}")
        return False


def notify_trade(event):
    """Send a formatted trade notification.

    event: timeline event dict with type, symbol, price, trade_pct, tag,
           trade_return, remaining_pct, etc.
    """
    sym = event["symbol"]
    typ = event["type"]
    tag = event.get("tag", "")
    pct = round(event.get("trade_pct", 0))
    price = event["price"]
    remaining = round(event.get("remaining_pct", 0))

    emoji = ":chart_with_upwards_trend:" if typ == "BUY" else ":chart_with_downwards_trend:"
    color = "#00c853" if typ == "BUY" else "#ff1744"

    title = f"{emoji} {typ} {pct}% {sym} ({tag})"
    fields = [f"*Price:* ${price:,.0f}", f"*Position:* {remaining}%"]

    if typ == "SELL" and event.get("trade_return") is not None:
        ret = event["trade_return"]
        fields.append(f"*Trade P&L:* {'+' if ret >= 0 else ''}{ret:.1f}%")

    text = f"{title}\n" + " | ".join(fields)

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": title},
        },
        {
            "type": "section",
            "fields": [{"type": "mrkdwn", "text": f} for f in fields],
        },
    ]

    return send_slack(text, blocks)
