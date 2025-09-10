import pandas as pd
import re
from pathlib import Path
from datetime import datetime, timezone

def clean_text(text):
    """
    Cleans text by removing HTML tags, keeping only alphanumeric and basic
    punctuation, and normalizing whitespace. This approach is more robust than
    listing specific junk characters.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags using a general pattern
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Keep only alphanumeric characters, spaces, and common punctuation.
    # This will remove all other special characters and garbled text.
    text = re.sub(r'[^a-zA-Z0-9\s.,?!-]', '', text)
    
    # 3. Replace all newline characters with a single space to prevent row merging
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 4. Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
raw_dir = BASE_DIR / "data" / "raw"
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Load latest raw file. Added `encoding='utf-8'` to correctly handle characters.
try:
    raw_file = max(raw_dir.glob("merged_with_sentiment_*.csv"), key=lambda f: f.stat().st_mtime)
    print(f"📂 Using raw file: {raw_file}")
    df = pd.read_csv(raw_file, encoding='utf-8')
except FileNotFoundError:
    print("❌ Error: No raw CSV file found. Please ensure your data is in the 'data/raw' folder.")
    exit()

# Apply the cleaning function to the 'title' and 'description' columns
df["title"] = df["title"].apply(clean_text)
df["description"] = df["description"].apply(clean_text)

# Drop any rows where the 'title' is empty, as this indicates a completely blank record.
df.dropna(subset=['title'], inplace=True)

# Convert 'publishedAt' to datetime (timezone-aware)
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)

# Fill missing dates with current UTC timestamp (timezone-aware)
now = datetime.now(timezone.utc)
df["publishedAt"] = df["publishedAt"].fillna(now)

# Convert datetime to string in desired format
df["publishedAt"] = df["publishedAt"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Save to processed folder. Added `encoding='utf-8'` to save correctly.
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
processed_file = processed_dir / f"cleaned_with_sentiment_{timestamp}.csv"
df.to_csv(processed_file, index=False, encoding='utf-8')

print(f"✅ Cleaned data saved to {processed_file} ({len(df)} rows)")


