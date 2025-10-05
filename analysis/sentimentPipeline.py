# sentimentPipeline.py
import pandas as pd
from pathlib import Path
from datetime import datetime
from analysis.sentimentAnalysis import analyze_sentiment  # uses sentimentAnalysis.py

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def run_sentiment_pipeline_on_latest():
    """
    Wrapper to run sentiment analysis on the latest cleaned_tech CSV.
    Returns the sentiment DataFrame.
    """
    return run_sentiment_pipeline()

def run_sentiment_pipeline():
    """
    Compute sentiment for all skills already present in the latest cleaned CSV.

    Returns:
        pd.DataFrame: ['skill','date','title','description','source','platform','text','sentiment','sentiment_score']
    """
    # -----------------------------
    # Load latest cleaned CSV
    # -----------------------------
    try:
        cleaned_file = max(PROCESSED_DIR.glob("cleaned_tech_*.csv"), key=lambda f: f.stat().st_mtime)
        print(f"📂 Using cleaned tech file: {cleaned_file}")
        df = pd.read_csv(cleaned_file, encoding="utf-8")
    except ValueError:
        print("❌ No cleaned CSV file found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        print("❌ Cleaned CSV is empty.")
        return pd.DataFrame()

    # -----------------------------
    # Ensure correct column names
    # -----------------------------
    if 'publishedAt' in df.columns:
        df.rename(columns={'publishedAt': 'date'}, inplace=True)
    if 'skill_query' in df.columns:
        df.rename(columns={'skill_query': 'skill'}, inplace=True)

    if 'date' not in df.columns:
        print("❌ CSV must contain 'date' column for forecasting.")
        return pd.DataFrame()
    if 'skill' not in df.columns:
        print("❌ CSV must contain 'skill' column for forecasting.")
        return pd.DataFrame()

    # Convert 'date' to datetime.date
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    # -----------------------------
    # Prepare text for sentiment
    # -----------------------------
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

    # -----------------------------
    # Apply sentiment analysis
    # -----------------------------
    df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(x)['label'])
    df['sentiment_score'] = df['text'].apply(lambda x: round(analyze_sentiment(x)['score'], 4))

    # -----------------------------
    # Save with timestamp
    # -----------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = PROCESSED_DIR / f"cleaned_with_sentiment_{timestamp}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n✅ Cleaned CSV with sentiments saved to {output_file} ({len(df)} rows)")
    # Uncomment below to debug
    # print("Sample rows:\n", df.head())

    return df

# -----------------------------
# Allow running standalone
# -----------------------------
if __name__ == "__main__":
    run_sentiment_pipeline()





    
  





