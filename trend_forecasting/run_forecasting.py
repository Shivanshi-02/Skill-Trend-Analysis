import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path
import glob
import os
import requests
from prophet import Prophet
from dotenv import load_dotenv # Import load_dotenv
import numpy as np

# -----------------------------
# 1️⃣ Forecast for a skill
# -----------------------------
def forecast_skill(daily_sentiment, skill, days_ahead=5):
    """
    Generates a forecast for a specific skill's sentiment trend using Prophet.

    Args:
        daily_sentiment (pd.DataFrame): DataFrame with daily aggregated sentiment data.
        skill (str): The skill to forecast.
        days_ahead (int): Number of days to forecast into the future.

    Returns:
        tuple: A tuple containing the history DataFrame and the forecast DataFrame.
    """
    df_skill = daily_sentiment[daily_sentiment['skill'] == skill].copy()
    if df_skill.empty:
        return None, None

    # Prepare data for Prophet: Prophet requires 'ds' (date) and 'y' (value) columns
    prophet_df = df_skill.rename(columns={'date': 'ds', 'sentiment_score': 'y'})
    
    # Initialize and fit the Prophet model
    m = Prophet()
    m.fit(prophet_df)
    
    # Create a future DataFrame to hold the forecast dates
    future = m.make_future_dataframe(periods=days_ahead)
    
    # Make the forecast
    forecast = m.predict(future)
    
    # The forecast data includes a 'yhat' column with the predicted values.
    # We'll use this to create our forecast_df.
    forecast_df = forecast[['ds', 'yhat']].tail(days_ahead).rename(
        columns={'ds': 'date', 'yhat': 'forecast_score'}
    )
    
    # Return the historical data and the forecast
    return df_skill, forecast_df

   

# -----------------------------
# 2️⃣ Detect anomalies
# -----------------------------
def detect_anomalies(df):
    """
    Detects anomalies in the sentiment data.

    An anomaly is defined as a point where the sentiment score deviates from the
    mean by more than 2 standard deviations.
    
    Args:
        df (pd.DataFrame): DataFrame with daily aggregated sentiment data.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows identified as anomalies.
    """
    mean = df['sentiment_score'].mean()
    std = df['sentiment_score'].std()
    anomalies = df[abs(df['sentiment_score'] - mean) > 2 * std].copy()
    return anomalies

# -----------------------------
# 3️⃣ Compare trends with all skills
# -----------------------------
def compare_trends(daily_sentiment, skills):
    """
    Compares the latest sentiment score of all skills.

    Args:
        daily_sentiment (pd.DataFrame): DataFrame with daily aggregated sentiment data.
        skills (list): A list of skills to compare.

    Returns:
        pd.DataFrame: A sorted DataFrame of the latest sentiment scores for the given skills.
    """
    latest = (
        daily_sentiment.sort_values('date')
        .groupby('skill')
        .tail(1)
        .sort_values('sentiment_score', ascending=False)
    )
    return latest[latest['skill'].isin(skills)]

# -----------------------------
# 4️⃣ Plot forecast
# -----------------------------
def plot_forecast(history_df, forecast_df, anomalies_df, skill, daily_sentiment):
    """
    Opens TWO SEPARATE matplotlib windows:
    1. Trend forecast for the selected skill
    2. Bar chart comparing sentiment scores of selected skill vs other skills
    """

    # ==============================
    # 1️⃣ Window 1: Trend Forecast
    # ==============================
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Actual
    plt.plot(history_df['date'], history_df['sentiment_score'],
             label="Actual", marker='o', color='#3b82f6', linewidth=2)

    # Forecast
    if forecast_df is not None and not forecast_df.empty:
        plt.plot(forecast_df['date'], forecast_df['forecast_score'],
                 label="5-Day Forecast", linestyle='--', marker='o',
                 color='#fb923c', linewidth=2)

    # Anomalies
    if anomalies_df is not None and not anomalies_df.empty:
        plt.scatter(anomalies_df['date'], anomalies_df['sentiment_score'],
                    color='#ef4444', label="Anomalies", s=100,
                    edgecolors='black', zorder=5)

    # Axis settings
    y_min = 0
    y_max = max(
        history_df['sentiment_score'].max(),
        forecast_df['forecast_score'].max() if forecast_df is not None else 0
    )
    y_max = np.ceil(y_max * 20) / 20
    y_ticks = np.arange(y_min, y_max + 0.05, 0.05)

    plt.ylim(y_min, y_max)
    plt.yticks(y_ticks)
    plt.title(f"Trend Forecast for '{skill}'", fontsize=16, fontweight='bold')
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Average Sentiment Score", fontsize=12)
    plt.xticks(rotation=30)
    plt.legend(frameon=True, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.show()   # ✅ Opens first window

    # ==============================
    # 2️⃣ Window 2: Skill Comparison Bar Chart
    # ==============================
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Prepare latest sentiment for all skills
    latest_sentiment = (
        daily_sentiment.groupby('skill')
        .tail(1)
        .sort_values('sentiment_score', ascending=False)
    )

    # Highlight selected skill
    bar_colors = [
        '#fb923c' if s == skill else '#3b82f6'
        for s in latest_sentiment['skill']
    ]

    plt.barh(latest_sentiment['skill'], latest_sentiment['sentiment_score'],
             color=bar_colors, edgecolor='black')

    plt.xlabel("Sentiment Score", fontsize=12)
    plt.ylabel("Skills", fontsize=12)
    plt.title(f"Sentiment Score Comparison\n({skill} vs Others)", fontsize=16)
    plt.tight_layout()
    plt.show()   # ✅ Opens second window
# -----------------------------
# 5️⃣ Main
# -----------------------------
def send_slack_alert(webhook_url, message):
    """
    Sends a message to a Slack channel using an incoming webhook.
    
    Args:
        webhook_url (str): The Slack webhook URL.
        message (str): The message text to send.
    """
    payload = {'text': message}
    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an exception for bad status codes
        print("✅ Slack alert sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Slack alert: {e}")

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Access the webhook URL from the environment
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    ALERT_THRESHOLD = 0.003 # You can adjust this value as needed

    # Pick latest CSV automatically
    folder_path = r"C:\Projects\skill-trend-analyzer\data\processed"
    list_of_files = glob.glob(os.path.join(folder_path, "cleaned_with_sentiment_*.csv"))
    if not list_of_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📂 Using latest CSV: {latest_file}")

    daily_sentiment = pd.read_csv(latest_file)
    
    # --- The crucial change: Aggregate data daily before processing ---
    daily_sentiment['publishedAt'] = pd.to_datetime(daily_sentiment['publishedAt']).dt.date
    daily_sentiment['skill'] = daily_sentiment['skill_query']
    
    # Group by date and skill, then calculate the mean sentiment score for each day
    daily_sentiment = daily_sentiment.groupby(['publishedAt', 'skill'])['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'publishedAt': 'date'}, inplace=True)
    
    # Correct skills.json path
    skills_json = Path(r"C:\Projects\skill-trend-analyzer\data_pipeline\skills.json")
    if not skills_json.exists():
        raise FileNotFoundError(f"skills.json not found at {skills_json}")
    
    skills_data = json.load(open(skills_json))
    # For now, pick all skills across categories
    skills = [skill for cat in skills_data.values() for skill in cat]

    # Ask user for skill
    skill_to_search = input(f"Enter a skill to forecast from {skills}: ").strip()
    if skill_to_search not in skills:
        print(f"⚠️ Skill '{skill_to_search}' not in skills.json")
        exit()

    # Forecast
    history_df, forecast_df = forecast_skill(daily_sentiment, skill_to_search)
    if history_df is None:
        print(f"No data available for '{skill_to_search}'")
        exit()

    # Anomalies
    anomalies_df = detect_anomalies(history_df)

    # Compare with other skills
    comparison = compare_trends(daily_sentiment, skills)
    print("\n🔥 Current Market Demand Ranking:")
   # Reset index for sequential numbering in terminal
    comparison_reset = comparison[['skill', 'sentiment_score']].reset_index(drop=True)
    print(comparison_reset)

    # Plot
    plot_forecast(history_df, forecast_df, anomalies_df, skill_to_search, daily_sentiment)


    # Save forecast
    out_path = Path(r"C:\Projects\skill-trend-analyzer\data\forecast")
    out_path.mkdir(parents=True, exist_ok=True)
    if forecast_df is not None:
      forecast_file = out_path / f"{skill_to_search.replace(' ', '_')}_forecast.csv"
      forecast_df.to_csv(forecast_file, index=False)
    print(f"\n💾 Forecast saved to: {forecast_file}")

    
    # Check for sentiment change and send a Slack alert
    # Make sure we have at least 2 days of data
    if len(daily_sentiment) >= 2:
        last_two_days = daily_sentiment[daily_sentiment['skill'] == skill_to_search].tail(2).sort_values('date')
        if len(last_two_days) == 2:
            # Calculate the change in sentiment score between the last two days
            sentiment_change = last_two_days['sentiment_score'].iloc[-1] - last_two_days['sentiment_score'].iloc[0]
            
            # Check if the change exceeds the alert threshold and if the URL is set
            if abs(sentiment_change) > ALERT_THRESHOLD and SLACK_WEBHOOK_URL:
                message = f"📢 Sentiment alert! Change of {sentiment_change:.3f} detected in daily average sentiment for {skill_to_search}."
                send_slack_alert(SLACK_WEBHOOK_URL, message)