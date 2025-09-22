import pandas as pd
import re
from pathlib import Path
from datetime import datetime, timezone

def clean_text(text):
    """Remove HTML tags, special chars, extra spaces and newlines."""
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9\s.,?!-]", "", text)
    # Replace newlines with space
    text = text.replace("\n", " ").replace("\r", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
raw_dir = BASE_DIR / "data" / "raw"
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Load latest RAW file (now without sentiment)
try:
    raw_file = max(raw_dir.glob("merged_raw_*.csv"), key=lambda f: f.stat().st_mtime)
    print(f"📂 Using raw file: {raw_file}")
    df = pd.read_csv(raw_file, encoding="utf-8")
except ValueError:
    print("❌ No raw CSV file found in data/raw.")
    exit()

# Clean text columns
df["title"] = df["title"].apply(clean_text)
df["description"] = df["description"].apply(clean_text)

# Drop rows where title is empty after cleaning
df.dropna(subset=["title"], inplace=True)

# Convert publishedAt to UTC datetime
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)

# Fill missing dates with current UTC time
now = datetime.now(timezone.utc)
df["publishedAt"] = df["publishedAt"].fillna(now)

# Convert datetime back to string
df["publishedAt"] = df["publishedAt"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Save cleaned file (NO sentiment yet)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
processed_file = processed_dir / f"cleaned_raw_{timestamp}.csv"
df.to_csv(processed_file, index=False, encoding="utf-8")

print(f"✅ Cleaned data saved to {processed_file} ({len(df)} rows)")



