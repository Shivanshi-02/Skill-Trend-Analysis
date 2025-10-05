# newsPipeline.py
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from requests.auth import HTTPBasicAuth
import re

# ----------------------------
# Setup directories
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "data" / "raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
NEWSAPI_TOKEN = os.getenv("NEWSAPI_TOKEN")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSDATA_TOKEN = os.getenv("NEWSDATA_TOKEN")
NEWSDATA_URL = "https://newsdata.io/api/1/news"
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
SERP_API_KEY = os.getenv("SERP_API_KEY")
SERP_API_URL = "https://serpapi.com/search"

# ----------------------------
# Tech Keywords for trending detection
# ----------------------------
TECH_KEYWORDS = [
    "Python", "JavaScript", "Java", "C++", "C#", "Rust", "Go",
    "React", "Angular", "Vue", "Django", "Flask", "Node.js",
    "TensorFlow", "PyTorch", "Keras", "Pandas", "NumPy",
    "Figma", "Canva", "Photoshop", "Docker", "Kubernetes",
    "AWS", "Azure", "GCP", "Linux", "Git", "SQL", "NoSQL"
]

# ----------------------------
# Fetch functions
# ----------------------------
def fetch_newsapi(query, max_results=20):
    url = f"{NEWSAPI_URL}?q={query}&apiKey={NEWSAPI_TOKEN}&language=en&pageSize={max_results}"
    try:
        response = requests.get(url).json()
    except Exception as e:
        print(f"Error fetching NewsAPI for '{query}': {e}")
        return pd.DataFrame()
    if response.get("status") != "ok":
        print(f"NewsAPI error for '{query}': {response.get('message')}")
        return pd.DataFrame()
    articles = []
    for a in response.get("articles", []):
        published_at = pd.to_datetime(a.get("publishedAt"), errors='coerce')
        if pd.isna(published_at):
            published_at = datetime.now()
        articles.append({
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": published_at.isoformat(),
            "platform": "NewsAPI"
        })
    return pd.DataFrame(articles)


def fetch_newsdata(query, max_results=20):
    params = {"apikey": NEWSDATA_TOKEN, "q": query, "language": "en"}
    try:
        response = requests.get(NEWSDATA_URL, params=params).json()
    except Exception as e:
        print(f"Error fetching NewsData.io for '{query}': {e}")
        return pd.DataFrame()
    articles = []
    for a in response.get("results", [])[:max_results]:
        published_at = pd.to_datetime(a.get("pubDate"), errors='coerce')
        if pd.isna(published_at):
            published_at = datetime.now()
        articles.append({
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source_id"),
            "publishedAt": published_at.isoformat(),
            "platform": "NewsData"
        })
    return pd.DataFrame(articles)


def fetch_googlenews(query, max_results=20):
    params = {"engine": "google_news", "q": query, "api_key": SERP_API_KEY}
    try:
        response = requests.get(SERP_API_URL, params=params).json()
    except Exception as e:
        print(f"Error fetching Google News for '{query}': {e}")
        return pd.DataFrame()
    articles = []
    for a in response.get("news_results", [])[:max_results]:
        published_at = pd.to_datetime(a.get("date"), errors='coerce')
        if pd.isna(published_at):
            published_at = datetime.now()
        source_val = a.get("source")
        if isinstance(source_val, dict):
            source_val = source_val.get("name")
        articles.append({
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("snippet"),
            "source": source_val,
            "publishedAt": published_at.isoformat(),
            "platform": "GoogleNews"
        })
    return pd.DataFrame(articles)


def get_reddit_token():
    auth = HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    data = {"grant_type": "password", "username": REDDIT_USERNAME, "password": REDDIT_PASSWORD}
    headers = {"User-Agent": "NewsFetcherBot/0.1"}
    try:
        res = requests.post("https://www.reddit.com/api/v1/access_token",
                            auth=auth, data=data, headers=headers)
        return res.json().get("access_token")
    except Exception as e:
        print(f"Error getting Reddit token: {e}")
        return None


def fetch_reddit(query, token, max_results=20):
    headers = {"Authorization": f"bearer {token}", "User-Agent": "NewsFetcherBot/0.1"}
    params = {"q": query, "limit": max_results, "sort": "new"}
    try:
        response = requests.get("https://oauth.reddit.com/search",
                                headers=headers, params=params).json()
    except Exception as e:
        print(f"Error fetching Reddit for '{query}': {e}")
        return pd.DataFrame()
    posts = []
    for p in response.get("data", {}).get("children", []):
        post = p["data"]
        published_at = datetime.fromtimestamp(post.get("created_utc")) if post.get("created_utc") else datetime.now()
        posts.append({
            "skill_query": query,
            "title": post.get("title"),
            "description": post.get("selftext"),
            "source": post.get("subreddit"),
            "publishedAt": published_at.isoformat(),
            "platform": "Reddit"
        })
    return pd.DataFrame(posts)


# ----------------------------
# Detect trending tech skills
# ----------------------------
def extract_trending_skills(df, top_n=5):
    text = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.lower().tolist()
    text_str = " ".join(text)
    skill_counts = {}
    for kw in TECH_KEYWORDS:
        pattern = re.compile(rf"\b{re.escape(kw.lower())}\b")
        skill_counts[kw] = len(pattern.findall(text_str))
    
    # Sort by count, descending
    sorted_skills = [k for k, v in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)]
    
    # Pick top_n skills even if count=0
    top_skills = sorted_skills[:top_n]
    return top_skills


# ----------------------------
# Main
# ----------------------------
def fetch_news_for_user_skill(user_skill):
    print(f"🚀 Running full pipeline for {user_skill} ...")

    reddit_token = get_reddit_token()
    if not reddit_token:
        print("⚠️ Reddit token failed. Skipping Reddit fetch.")

    # Step 1: Fetch general technology news to find trending skills
    tech_news = pd.DataFrame()
    for func in [fetch_newsapi, fetch_newsdata, fetch_googlenews]:
        tech_news = pd.concat([tech_news, func("technology")], ignore_index=True)
    if reddit_token:
        tech_news = pd.concat([tech_news, fetch_reddit("technology", reddit_token)], ignore_index=True)

    trending_skills = extract_trending_skills(tech_news, top_n=4)
    final_topics = list(dict.fromkeys([user_skill] + trending_skills))  # user skill + top 4 trending
    print(f"📌 Final topics to fetch news for: {final_topics}")

    # Step 2: Fetch news for all 5 topics
    all_data = pd.DataFrame(columns=["skill_query","title","description","source","publishedAt","platform"])
    for i, skill in enumerate(final_topics, 1):
        print(f"[{i}/{len(final_topics)}] Fetching news for '{skill}' from all sources...")
        all_data = pd.concat([all_data, fetch_newsapi(skill)], ignore_index=True)
        all_data = pd.concat([all_data, fetch_newsdata(skill)], ignore_index=True)
        all_data = pd.concat([all_data, fetch_googlenews(skill)], ignore_index=True)
        if reddit_token:
            all_data = pd.concat([all_data, fetch_reddit(skill, reddit_token)], ignore_index=True)
        time.sleep(1)

    # Save raw CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = OUTPUT_DIR / f"merged_raw_{timestamp}.csv"
    all_data.to_csv(output_file, index=False)
    print(f"\n✅ Raw data saved to {output_file} ({len(all_data)} rows) - includes all user + top 4 trending skills")

    return output_file  # return path for further processing


