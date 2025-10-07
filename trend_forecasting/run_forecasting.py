# trend_forecasting/run_forecasting.py
import os
import json
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from analysis.dashboard import StrategicDashboard

# Ensure environment loaded (useful when run under Streamlit)
load_dotenv()

# Config (can be overridden by environment variables)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.003"))

def _mask_webhook(url):
    if not url:
        return None
    return url[:30] + "..." + url[-6:]

# -----------------------------
# Slack helper (top-level, importable)
# -----------------------------
def send_slack_alert(webhook_url, message, timeout=10):
    """
    Send a JSON message to Slack webhook.
    Returns (True, response_text) on success, (False, error_str) on failure.
    """
    if not webhook_url:
        print("⚠️ send_slack_alert: webhook_url is None")
        return False, "no-webhook"

    payload = {"text": message}
    try:
        resp = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        status = resp.status_code
        text = resp.text
        if 200 <= status < 300:
            print(f"✅ Slack POST OK ({status}) — resp: {text}")
            return True, f"{status}:{text}"
        else:
            print(f"❌ Slack POST failed ({status}) — resp: {text}")
            return False, f"{status}:{text}"
    except Exception as e:
        print(f"❌ Exception sending Slack alert: {e}")
        return False, str(e)

# -----------------------------
# Alert checker: compare yesterday vs today per skill
# -----------------------------
def check_and_alert_sentiment_change(daily_sentiment_all,
                                     webhook_url=SLACK_WEBHOOK_URL,
                                     alert_threshold=ALERT_THRESHOLD):
    """
    For each skill in daily_sentiment_all, compare the last two dates' sentiment_score.
    Sends a Slack message when change >= alert_threshold (or always for first data point).
    Returns a list of results dictionaries: {'skill','change','sent','detail'}.
    """
    results = []

    if daily_sentiment_all is None or daily_sentiment_all.empty:
        print("⚠️ check_and_alert_sentiment_change: no data -> nothing to do")
        return results

    required = {"date", "skill", "sentiment_score"}
    if not required.issubset(set(daily_sentiment_all.columns)):
        print(f"❌ check_and_alert_sentiment_change: missing columns. found: {daily_sentiment_all.columns.tolist()}")
        return results

    # Ensure date is datetime
    daily = daily_sentiment_all.copy()
    try:
        daily['date'] = pd.to_datetime(daily['date'])
    except Exception as e:
        print("❌ Failed converting date:", e)

    all_skills = daily['skill'].unique()
    print(f"🔔 Running alerts for {len(all_skills)} skills. webhook={_mask_webhook(webhook_url)}, threshold={alert_threshold}")

    for skill in all_skills:
        skill_history = daily[daily['skill'] == skill].sort_values('date')
        sent = False
        detail = ""
        change = None

        if skill_history.empty:
            detail = "no-data"
        else:
            if len(skill_history) >= 2:
                change = float(skill_history['sentiment_score'].iloc[-1] - skill_history['sentiment_score'].iloc[-2])
            else:
                # only 1 day of data -> treat as first datapoint with change 0.0
                change = 0.0

            # Decide to send based on threshold or first datapoint
            if abs(change) >= float(alert_threshold) or len(skill_history) == 1:
                message = f":loudspeaker: SENTIMENT ALERT: Change {change:+.3f} detected for **{skill}**."
                print(f"→ Attempting to send alert for {skill}: change={change:+.3f}")
                ok, info = send_slack_alert(webhook_url, message)
                sent = ok
                detail = info
            else:
                detail = f"change_below_threshold:{change:+.3f}"

        results.append({
            "skill": skill,
            "change": None if change is None else round(change, 6),
          #  "sent": sent,
        # "detail": detail
        })

    return results

# -----------------------------
# Main runner (top-level, importable)
# -----------------------------
def run_forecasting_and_alerts():
    """
    Loads latest cleaned sentiment via StrategicDashboard,
    triggers Slack alerts for each skill, prints forecasting/anomaly summaries,
    and returns alert results list.
    """
    dashboard = StrategicDashboard()
    if dashboard.daily_sentiment is None or dashboard.daily_sentiment.empty:
        print("❌ No sentiment data found. Run sentiment pipeline first.")
        return []

    # dashboard.daily_sentiment expected to be daily aggregated DataFrame with columns date, skill, sentiment_score
    daily_df = dashboard.daily_sentiment.copy()

    # If daily_df doesn't have sentiment_score (edge cases), attempt to reconstruct from cleaned_with_sentiment file
    if not set(['date', 'skill', 'sentiment_score']).issubset(set(daily_df.columns)):
        try:
            proc_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
            latest = max(list(proc_dir.glob("cleaned_with_sentiment_*.csv")), key=lambda f: f.stat().st_mtime)
            raw = pd.read_csv(latest)
            if 'publishedAt' in raw.columns:
                raw.rename(columns={'publishedAt': 'date'}, inplace=True)
            if 'skill_query' in raw.columns:
                raw.rename(columns={'skill_query': 'skill'}, inplace=True)
            raw['date'] = pd.to_datetime(raw['date'])
            daily_df = raw.groupby(['date', 'skill'])['sentiment_score'].mean().reset_index()
            print("ℹ️ Rebuilt daily_df from cleaned_with_sentiment file.")
        except Exception as e:
            print("❌ Could not rebuild daily_df:", e)
            return []

    # 1️⃣ Slack alerts
    results = check_and_alert_sentiment_change(daily_df, webhook_url=SLACK_WEBHOOK_URL, alert_threshold=ALERT_THRESHOLD)

    # 2️⃣ Forecast & anomalies (prints only)
    print("📈 Running forecasting (prints) for skills:")
    for skill in daily_df['skill'].unique():
        history_df, forecast_df = dashboard.forecast_skill(skill)
        anomalies = dashboard.detect_anomalies(history_df)
        if not anomalies.empty:
            print(f"⚠️ Anomalies detected for {skill} ({len(anomalies)} points)")
        else:
            print(f"✅ {skill}: No anomalies found.")

    return results 