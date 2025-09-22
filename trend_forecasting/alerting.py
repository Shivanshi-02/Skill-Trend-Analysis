# trend_forecasting/alerting.py
import pandas as pd
from .slack_notifier import send_slack_alert

def check_alerts(forecast_df, daily_sentiment_grouped, daily_keywords, 
                 sentiment_threshold=0.001, keyword_jump_pct=0.5):       #tweak threshold for alertssss
    """
    Checks for significant changes in sentiment or keyword trends and sends Slack alerts.
    
    Parameters:
    - forecast_df: DataFrame from Prophet containing forecasted sentiment ('yhat')
    - daily_sentiment_grouped: DataFrame with daily average sentiment ('daily_avg_sentiment')
    - daily_keywords: DataFrame or dict with daily keyword counts
    - sentiment_threshold: float, minimum change in sentiment to trigger alert
    - keyword_jump_pct: float, minimum percentage jump in keyword count to trigger alert
    """
    alerts_sent = []

    # Debug: check columns
    print("Columns in daily_sentiment_grouped:", daily_sentiment_grouped.columns.tolist())
    print("Columns in forecast_df:", forecast_df.columns.tolist())

    # --- Sentiment Alert ---
    # Compare yesterday vs day before yesterday
    if "daily_avg_sentiment" in daily_sentiment_grouped.columns:
        today_sentiment = daily_sentiment_grouped["daily_avg_sentiment"].iloc[-1]
        prev_sentiment = daily_sentiment_grouped["daily_avg_sentiment"].iloc[-2]

        change = today_sentiment - prev_sentiment
        if abs(change) >= sentiment_threshold:
            msg = f"⚠️ Sentiment alert! Change of {change:.3f} detected in daily average sentiment."
            send_slack_alert(msg)
            alerts_sent.append(msg)

    # --- Keyword Alert ---
    if daily_keywords is not None and len(daily_keywords) >= 2:
        last = daily_keywords.iloc[-1]["count"]
        prev = daily_keywords.iloc[-2]["count"]
        if prev > 0 and (last - prev) / prev >= keyword_jump_pct:
            msg = f"⚠️ Keyword surge alert! Keyword count jumped by {(last - prev) / prev:.2%}."
            send_slack_alert(msg)
            alerts_sent.append(msg)

    if not alerts_sent:
        print("✅ No alerts triggered today.")

    return alerts_sent
