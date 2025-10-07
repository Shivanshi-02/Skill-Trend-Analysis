import pandas as pd
from pathlib import Path
from datetime import datetime
from analysis.sentimentAnalysis import analyze_sentiment
import os
import json
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
def run_sentiment_pipeline():
    """
    Compute sentiment for all skills already present in the latest cleaned CSV.
    """
    try:
        cleaned_file = max(PROCESSED_DIR.glob("cleaned_tech_*.csv"), key=lambda f: f.stat().st_mtime)
        print(f"📂 Using cleaned tech file: {cleaned_file}")
        df = pd.read_csv(cleaned_file, encoding="utf-8")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Ensure column names
    if 'publishedAt' in df.columns:
        df.rename(columns={'publishedAt': 'date'}, inplace=True)
    if 'skill_query' in df.columns:
        df.rename(columns={'skill': 'skill'}, inplace=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

    # Apply sentiment
    df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(x)['label'])
    df['sentiment_score'] = df['text'].apply(lambda x: round(analyze_sentiment(x)['score'],4))

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = PROCESSED_DIR / f"cleaned_with_sentiment_{timestamp}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Cleaned CSV with sentiments saved to {output_file} ({len(df)} rows)")
    return df

# -----------------------------
def run_sentiment_pipeline_on_latest(clean_only=False):
    """
    Runs sentiment analysis on the latest cleaned CSV.
    If clean_only=True, it skips fetching news / regenerating cleaned CSV
    """
    try:
        cleaned_file = max(PROCESSED_DIR.glob("cleaned_tech_*.csv"), key=lambda f: f.stat().st_mtime)
        print(f"📂 Using latest cleaned tech file: {cleaned_file}")
        df = pd.read_csv(cleaned_file, encoding="utf-8")
    except Exception as e:
        print(f"❌ Failed to load cleaned CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Ensure columns
    if 'publishedAt' in df.columns:
        df.rename(columns={'publishedAt':'date'}, inplace=True)
    if 'skill_query' in df.columns:
        df.rename(columns={'skill_query':'skill'}, inplace=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

    # Only run sentiment if not already present
    if 'sentiment_score' not in df.columns or 'sentiment' not in df.columns:
        df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(x)['label'])
        df['sentiment_score'] = df['text'].apply(lambda x: round(analyze_sentiment(x)['score'],4))
        # Save CSV with sentiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = PROCESSED_DIR / f"cleaned_with_sentiment_{timestamp}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"✅ Saved sentiment CSV: {output_file}")

    # Daily sentiment aggregation
    daily_sentiment = (
        df.groupby(['date','skill'])['sentiment_score']
        .mean()
        .reset_index()
        .sort_values(['skill','date'])
    )
    return daily_sentiment

# -----------------------------
def send_slack_alerts_for_skills(daily_sentiment, alert_threshold=0.001, webhook_url=None):
    if webhook_url is None:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL not set")
        return

    for skill in daily_sentiment['skill'].unique():
        skill_data = daily_sentiment[daily_sentiment['skill']==skill].sort_values('date').tail(2)
        if len(skill_data) == 2:
            change = skill_data['sentiment_score'].iloc[-1] - skill_data['sentiment_score'].iloc[0]
            if abs(change) >= alert_threshold:
                message = f":loudspeaker: SENTIMENT ALERT: Change {change:+.3f} detected for **{skill}**."
                payload = {'text': message}
                try:
                    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type':'application/json'})
                except Exception as e:
                    print(f"❌ Failed to send Slack alert: {e}")

# -----------------------------
if __name__ == "__main__":
    run_sentiment_pipeline()



    
  





