import os
import pandas as pd

def get_latest_sentiment_csv():
    """
    Find the latest cleaned_with_sentiment CSV inside data/processed.
    This avoids manually renaming files each time a new batch is generated.
    """
    folder = "data/processed"
    files = [f for f in os.listdir(folder) if f.startswith("cleaned_with_sentiment")]
    if not files:
        raise FileNotFoundError("No cleaned_with_sentiment CSV found in data/processed/")
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return os.path.join(folder, latest)

def load_and_prepare_data():
    """
    Load the latest sentiment CSV and create:
      1. daily_sentiment: Daily average sentiment for forecasting
      2. daily_keywords: Daily keyword/article count for surge detection
    """
    file_path = get_latest_sentiment_csv()
    print(f"📂 Using file: {os.path.basename(file_path)}")

    # Read CSV and parse publishedAt as datetime
    df = pd.read_csv(file_path, parse_dates=["publishedAt"])

    # ---- Daily Sentiment ----
    daily_sentiment = (
        df.groupby(df["publishedAt"].dt.floor("D"))["sentiment_score"]
          .mean()
          .reset_index()
          .rename(columns={"publishedAt": "ds", "sentiment_score": "y"})
    )

    # ---- Daily Keyword/Article Count ----
    # Use .size() to get numeric counts and keep Timestamp (not Python date)
    daily_keywords = (
        df.groupby(df["publishedAt"].dt.floor("D"))
          .size()
          .reset_index(name="count")
          .rename(columns={"publishedAt": "date"})
    )

    return daily_sentiment, daily_keywords



