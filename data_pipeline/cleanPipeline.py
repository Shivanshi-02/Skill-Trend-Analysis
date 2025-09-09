import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
raw_dir = BASE_DIR / "data" / "raw"
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Load latest raw file
raw_file = max(raw_dir.glob("merged_with_sentiment_*.csv"), key=lambda f: f.stat().st_mtime)
print(f"📂 Using raw file: {raw_file}")

df = pd.read_csv(raw_file)

# Convert 'publishedAt' to datetime (timezone-aware)
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)

# Fill missing dates with current UTC timestamp (timezone-aware)
now = datetime.now(timezone.utc)
df["publishedAt"] = df["publishedAt"].fillna(now)

# Convert datetime to string in desired format
df["publishedAt"] = df["publishedAt"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Save to processed folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
processed_file = processed_dir / f"cleaned_with_sentiment_{timestamp}.csv"
df.to_csv(processed_file, index=False)

print(f"✅ Cleaned data saved to {processed_file} ({len(df)} rows)")


