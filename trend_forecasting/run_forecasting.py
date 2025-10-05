# run_forecasting.py
import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from analysis.sentimentPipeline import run_sentiment_pipeline
except ImportError:
    pass

# -----------------------------
# Helper functions
# -----------------------------
def forecast_skill(daily_sentiment, skill, days_ahead=5):
    from prophet import Prophet
    df_skill = daily_sentiment[daily_sentiment['skill'] == skill].copy()
    df_skill['date'] = pd.to_datetime(df_skill['date'])
    
    if df_skill.empty:
        return None, None
    
    prophet_df = df_skill.rename(columns={'date': 'ds', 'sentiment_score': 'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=days_ahead)
    forecast = m.predict(future)
    forecast_df = forecast[['ds', 'yhat']].tail(days_ahead).rename(columns={'ds': 'date', 'yhat': 'forecast_score'})
    return df_skill, forecast_df

def detect_anomalies(df):
    if df.empty:
        return pd.DataFrame()
    mean = df['sentiment_score'].mean()
    std = df['sentiment_score'].std()
    anomalies = df[abs(df['sentiment_score'] - mean) > 2 * std].copy()
    return anomalies

def send_slack_alert(webhook_url, message):
    payload = {'text': message}
    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        print("✅ Slack alert sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Slack alert: {e}")

def check_and_alert_sentiment_change(daily_sentiment_all, skill_list, alert_threshold, webhook_url):
    if not webhook_url:
        return
    for skill in skill_list:
        skill_history = daily_sentiment_all[daily_sentiment_all['skill'] == skill].sort_values('date').tail(2)
        if len(skill_history) == 2:
            change = skill_history['sentiment_score'].iloc[-1] - skill_history['sentiment_score'].iloc[0]
            if abs(change) > alert_threshold:
                message = f"📢 SENTIMENT ALERT: Change {change:+.3f} detected for **{skill}**."
                send_slack_alert(webhook_url, message)

# -----------------------------
# Plot functions
# -----------------------------
def plot_comparison_bar(daily_sentiment, user_skill):
    latest_date = daily_sentiment['date'].max()
    latest_sentiment = daily_sentiment[daily_sentiment['date'] == latest_date].copy()
    if latest_sentiment.empty:
        return
    sorted_skills = latest_sentiment.sort_values('sentiment_score', ascending=False)
    user_row = sorted_skills[sorted_skills['skill'] == user_skill]
    if user_row.empty:
        return
    top_4_others = sorted_skills[sorted_skills['skill'] != user_skill].head(4)
    top5 = pd.concat([user_row, top_4_others]).drop_duplicates(subset=['skill'])
    top5 = top5.sort_values('sentiment_score', ascending=False)
    plt.figure(figsize=(8,6))
    colors = ['#fb923c' if s==user_skill else '#3b82f6' for s in top5['skill']]
    plt.bar(top5['skill'], top5['sentiment_score'], color=colors, edgecolor='black')
    plt.title(f"Sentiment Comparison: '{user_skill}' vs Top 4", fontsize=14, fontweight='bold')
    plt.xlabel("Skills", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def plot_user_forecast(history_df, forecast_df, anomalies_df, user_skill):
    plt.figure(figsize=(12,6))
    history_df['date'] = pd.to_datetime(history_df['date'])
    if forecast_df is not None:
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    if anomalies_df is not None and not anomalies_df.empty:
        anomalies_df['date'] = pd.to_datetime(anomalies_df['date'])
    plt.plot(history_df['date'], history_df['sentiment_score'], marker='o', color='#3b82f6', linewidth=2, label='Actual (Daily Avg)', zorder=2)
    if forecast_df is not None and not forecast_df.empty:
        plt.plot(forecast_df['date'], forecast_df['forecast_score'], linestyle='--', marker='o', color='#fb923c', linewidth=2, label='5-Day Forecast', zorder=2)
    if anomalies_df is not None and not anomalies_df.empty:
        plt.scatter(anomalies_df['date'], anomalies_df['sentiment_score'], color='#ef4444', s=100, edgecolors='black', label=r"Anomalies (2$\sigma$)", zorder=5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.title(f"Sentiment Trend & Forecast for '{user_skill}'", fontsize=16, fontweight='bold')
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Average Sentiment Score", fontsize=12)
    plt.xticks(rotation=30)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    ALERT_THRESHOLD = 0.003

    PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    list_of_files = glob.glob(str(PROCESSED_DIR / "cleaned_with_sentiment*.csv"))
    if not list_of_files:
        print(f"❌ No sentiment CSV files found. Run sentiment pipeline first.")
        return
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📂 Using sentiment CSV: {latest_file}")
    raw_sentiment_data = pd.read_csv(latest_file)

    # -----------------------------
    # Ensure date column exists
    # -----------------------------
    if 'date' not in raw_sentiment_data.columns:
        print("❌ CSV must contain 'date' column.")
        return
    raw_sentiment_data['date'] = pd.to_datetime(raw_sentiment_data['date']).dt.date

    # Daily aggregation
    daily_sentiment_all = (
        raw_sentiment_data.groupby(['date','skill'])['sentiment_score']
        .mean()
        .reset_index()
    )

    # --- User skill input ---
    USER_INPUT_SKILL_RAW = "Rust"  # replace with dynamic input
    available_skills = daily_sentiment_all['skill'].unique()
    user_skill_cased = next((s for s in available_skills if s.lower() == USER_INPUT_SKILL_RAW.lower()), None)
    if user_skill_cased is None:
        print(f"❌ Skill '{USER_INPUT_SKILL_RAW}' not found.")
        return
    user_skill = user_skill_cased
    print(f"🎯 Analyzing for user skill: {user_skill}")

    # --- Top-4 comparison skills
    latest_date = daily_sentiment_all['date'].max()
    latest_sentiment = daily_sentiment_all[daily_sentiment_all['date'] == latest_date].copy()
    sorted_skills = latest_sentiment.sort_values('sentiment_score', ascending=False)
    top_4_others = sorted_skills[sorted_skills['skill'] != user_skill].head(4)['skill'].tolist()
    alert_skill_list = list(set([user_skill] + top_4_others))

    check_and_alert_sentiment_change(daily_sentiment_all, alert_skill_list, ALERT_THRESHOLD, SLACK_WEBHOOK_URL)

    # Forecast & anomalies
    history_df_prophet, forecast_df = forecast_skill(daily_sentiment_all, user_skill)
    anomalies_df = detect_anomalies(history_df_prophet)

    # Plots
    plot_comparison_bar(daily_sentiment_all, user_skill)
    plot_user_forecast(history_df_prophet, forecast_df, anomalies_df, user_skill)

    # Save forecast CSV
    FORECAST_DIR = Path(__file__).resolve().parent.parent / "data" / "forecast"
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    if forecast_df is not None:
        forecast_file = FORECAST_DIR / f"{user_skill.replace(' ','_')}_forecast.csv"
        forecast_df.to_csv(forecast_file, index=False)
        print(f"💾 Forecast saved to: {forecast_file}")

if __name__ == "__main__":
    main()


