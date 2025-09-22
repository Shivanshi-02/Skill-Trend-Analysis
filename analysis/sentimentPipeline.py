import pandas as pd
from pathlib import Path
from datetime import datetime
from .sentimentAnalysis import analyze_sentiment #import from same package

BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Load latest cleaned file
clean_file = max(processed_dir.glob("cleaned_raw_*.csv"), key=lambda f: f.stat().st_mtime)
print(f"📂 Using cleaned file: {clean_file}")
df = pd.read_csv(clean_file, encoding="utf-8")

# Run sentiment analysis on combined title + description
results = df.apply(
    lambda row: analyze_sentiment(f"{row['title']} {row['description']}"),
    axis=1
)
df["sentiment"] = results.apply(lambda x: x["label"])
df["sentiment_score"] = results.apply(lambda x: x["score"])

# Save final output with sentiment
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
out_file = processed_dir / f"cleaned_with_sentiment_{timestamp}.csv"
df.to_csv(out_file, index=False, encoding="utf-8")

print(f"✅ Sentiment data saved to {out_file} ({len(df)} rows)")
