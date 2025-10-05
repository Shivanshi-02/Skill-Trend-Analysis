# cleanPipeline.py (Tech-only filter using Sentence-BERT)
import pandas as pd
import re
from pathlib import Path
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML
    text = re.sub(r"[^\w\s.,?!-:/]", "", text) 
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Semantic Tech Filter
# -----------------------------
TECH_SEEDS = [
    "technology", "programming", "software", "coding", 
    "computer", "AI", "machine learning", "data science", "engineering", "IT"
]

SIM_THRESHOLD = 0.2  # Similarity threshold for tech relevance

print("📌 Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded.")

def filter_tech_news(df, threshold=SIM_THRESHOLD):
    # Combine title + description
    df['text_for_classification'] = df['title'].fillna('') + ". " + df['description'].fillna('')
    
    # Compute embeddings
    tech_emb = model.encode(TECH_SEEDS, convert_to_tensor=True)
    news_emb = model.encode(df['text_for_classification'].tolist(), convert_to_tensor=True)
    
    # Compute max cosine similarity against tech seeds
    cosine_scores = util.cos_sim(news_emb, tech_emb)  # shape: (num_news, num_seeds)
    max_scores = cosine_scores.max(dim=1).values.cpu().numpy()
    
    df['tech_score'] = max_scores
    filtered_df = df[df['tech_score'] >= threshold].copy()
    
    # -----------------------------
    # Ensure every skill from raw CSV is represented at least once
    # -----------------------------
    for skill in df['skill_query'].unique():
        if skill not in filtered_df['skill_query'].unique():
            top_article = df[df['skill_query'] == skill].iloc[[0]]
            filtered_df = pd.concat([filtered_df, top_article], ignore_index=True)
    
    # Drop temp columns
    filtered_df.drop(columns=['text_for_classification', 'tech_score'], inplace=True, errors='ignore')
    
    return filtered_df

# -----------------------------
# Main Cleaning Pipeline
# -----------------------------
def run_clean_pipeline():
    # Load latest raw CSV
    try:
        raw_file = max(RAW_DIR.glob("merged_raw_*.csv"), key=lambda f: f.stat().st_mtime)
        print(f"📂 Using raw file: {raw_file}")
        df = pd.read_csv(raw_file, encoding='utf-8')
    except ValueError:
        print("❌ No raw CSV file found.")
        return
    except Exception as e:
        print("❌ Error reading raw CSV:", e)
        return
    
    if df.empty:
        print("❌ Raw CSV is empty.")
        return

    # Clean text columns
    df['title'] = df['title'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)

    initial_count = len(df)
    df = filter_tech_news(df)
    filtered_count = len(df)

    print(f"✅ Filter complete: {filtered_count}/{initial_count} tech-related articles kept. All skills preserved.")

    # Ensure publishedAt is parsed correctly
    now = datetime.now(timezone.utc)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce', utc=True).fillna(now)
    df['publishedAt'] = df['publishedAt'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save cleaned CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    cleaned_file = PROCESSED_DIR / f"cleaned_tech_{timestamp}.csv"
    df.to_csv(cleaned_file, index=False, encoding='utf-8')
    print(f"✅ Cleaned tech CSV saved: {cleaned_file} ({filtered_count} rows)")

if __name__ == "__main__":
    run_clean_pipeline()







