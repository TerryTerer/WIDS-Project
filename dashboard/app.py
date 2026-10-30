from pathlib import Path
import csv
from datetime import datetime

from flask import Flask, render_template, jsonify


BASE_DIR = Path(__file__).resolve().parent.parent
ALERT_LOG = BASE_DIR / "reports" / "alerts.csv"

app = Flask(__name__)


def load_alerts():
    """Load alerts from the detector's CSV log."""
    if not ALERT_LOG.exists():
        return []

    try:
        with ALERT_LOG.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)
    except (OSError, csv.Error):
        return []


def get_statistics(alerts):
    """Calculate dashboard statistics."""
    total_alerts = len(alerts)

    deauth_alerts = sum(
        1
        for alert in alerts
        if alert.get("attack_type", "").lower() == "deauthentication"
    )

    unknown_alerts = sum(
        1
        for alert in alerts
        if alert.get("attack_type", "").lower() == "unknown attack"
    )

    return {
        "total_alerts": total_alerts,
        "deauth_alerts": deauth_alerts,
        "unknown_alerts": unknown_alerts,
    }


@app.route("/")
def index():
    alerts = load_alerts()
    statistics = get_statistics(alerts)

    # Show newest alerts first.
    recent_alerts = list(reversed(alerts[-20:]))

    return render_template(
        "index.html",
        alerts=recent_alerts,
        statistics=statistics,
    )


@app.route("/api/alerts")
def api_alerts():
    """Return alerts as JSON for dashboard refreshes."""
    alerts = load_alerts()

    return jsonify(
        {
            "alerts": alerts[-50:],
            "statistics": get_statistics(alerts),
        }
    )


@app.route("/api/status")
def api_status():
    """Return basic WIDS system status."""
    return jsonify(
        {
            "status": "ONLINE",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "alert_log": str(ALERT_LOG),
            "alert_log_exists": ALERT_LOG.exists(),
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("WIDS WEB DASHBOARD")
    print("=" * 60)
    print(f"Alert log: {ALERT_LOG}")
    print("Dashboard: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
