import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import requests
import time
from requests.auth import HTTPBasicAuth

from analysis.sentimentAnalysis import analyze_sentiment
from pathlib import Path

# ----------------------------
# Find path relative to this file
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
SKILLS_FILE = BASE_DIR / "skills.json"

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()

# Credentials
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
# Load domains & skills from JSON
# ----------------------------
with open(SKILLS_FILE) as f:
    DOMAINS_AND_SKILLS = json.load(f)

# ----------------------------
# Fetch news articles (NewsAPI)
# ----------------------------
def fetch_newsapi(query, domain, max_results=20):
    url = f"{NEWSAPI_URL}?q={query}&apiKey={NEWSAPI_TOKEN}&language=en&pageSize={max_results}"
    try:
        response = requests.get(url).json()
    except Exception as e:
        print(f"Error fetching NewsAPI for '{query}': {e}")
        return pd.DataFrame(columns=["domain","skill_query","title","description","source","publishedAt","url","platform","sentiment","sentiment_score"])
    
    if response.get("status") != "ok":
        print(f"NewsAPI error for '{query}': {response.get('message')}")
        return pd.DataFrame(columns=["domain","skill_query","title","description","source","publishedAt","url","platform","sentiment","sentiment_score"])
    
    articles = []
    for a in response.get("articles", []):
        published_at = pd.to_datetime(a.get("publishedAt"), errors='coerce')
        if published_at is pd.NaT:
            published_at = datetime.now()

        text = f"{a.get('title', '')} {a.get('description', '')}"
        sentiment = analyze_sentiment(text)

        articles.append({
            "domain": domain,
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": published_at.isoformat(),
            "url": a.get("url"),
            "platform": "NewsAPI",
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"]
        })
    return pd.DataFrame(articles)

# ----------------------------
# Fetch news (NewsData.io)
# ----------------------------
def fetch_newsdata(query, domain, max_results=20):
    params = {"apikey": NEWSDATA_TOKEN, "q": query, "language": "en"}
    try:
        response = requests.get(NEWSDATA_URL, params=params).json()
    except Exception as e:
        print(f"Error fetching NewsData.io for '{query}': {e}")
        return pd.DataFrame(columns=["domain","skill_query","title","description","source","publishedAt","url","platform","sentiment","sentiment_score"])
    
    articles = []
    for a in response.get("results", [])[:max_results]:
        published_at = pd.to_datetime(a.get("pubDate"), errors='coerce')
        if published_at is pd.NaT:
            published_at = datetime.now()

        text = f"{a.get('title', '')} {a.get('description', '')}"
        sentiment = analyze_sentiment(text)

        articles.append({
            "domain": domain,
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source_id"),
            "publishedAt": published_at.isoformat(),
            "url": a.get("link"),
            "platform": "NewsData",
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"]
        })
    return pd.DataFrame(articles)

# ----------------------------
# Fetch news (Google News via SerpAPI)
# ----------------------------
def fetch_googlenews(query, domain, max_results=20):
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": SERP_API_KEY
    }
    try:
        response = requests.get(SERP_API_URL, params=params).json()
    except Exception as e:
        print(f"Error fetching Google News for '{query}': {e}")
        return pd.DataFrame(columns=["domain","skill_query","title","description","source","publishedAt","url","platform","sentiment","sentiment_score"])
    
    articles = []
    for a in response.get("news_results", [])[:max_results]:
        published_at = pd.to_datetime(a.get("date"), errors='coerce')
        if published_at is pd.NaT:
            published_at = datetime.now()

        text = f"{a.get('title', '')} {a.get('snippet', '')}"
        sentiment = analyze_sentiment(text)

        source_val = a.get("source")
        if isinstance(source_val, dict):
            source_val = source_val.get("name")

        articles.append({
            "domain": domain,
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("snippet"),
            "source": source_val,
            "publishedAt": published_at.isoformat(),
            "url": a.get("link"),
            "platform": "GoogleNews",
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"]
        })
    return pd.DataFrame(articles)

# ----------------------------
# Get Reddit Token
# ----------------------------
def get_reddit_token():
    auth = HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    data = {"grant_type": "password", "username": REDDIT_USERNAME, "password": REDDIT_PASSWORD}
    headers = {"User-Agent": "NewsFetcherBot/0.1"}
    try:
        res = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
        token = res.json().get("access_token")
        return token
    except Exception as e:
        print(f"Error getting Reddit token: {e}")
        return None

# ----------------------------
# Fetch Reddit posts
# ----------------------------
def fetch_reddit(query, domain, token, max_results=20):
    headers = {"Authorization": f"bearer {token}", "User-Agent": "NewsFetcherBot/0.1"}
    params = {"q": query, "limit": max_results, "sort": "new"}
    try:
        response = requests.get("https://oauth.reddit.com/search", headers=headers, params=params).json()
    except Exception as e:
        print(f"Error fetching Reddit for '{query}': {e}")
        return pd.DataFrame(columns=["domain","skill_query","title","description","source","publishedAt","url","platform","sentiment","sentiment_score"])
    
    posts = []
    for p in response.get("data", {}).get("children", []):
        post = p["data"]
        published_at = datetime.fromtimestamp(post.get("created_utc")) if post.get("created_utc") else datetime.now()

        text = f"{post.get('title', '')} {post.get('selftext', '')}"
        sentiment = analyze_sentiment(text)

        posts.append({
            "domain": domain,
            "skill_query": query,
            "title": post.get("title"),
            "description": post.get("selftext"),
            "source": post.get("subreddit"),
            "publishedAt": published_at.isoformat(),
            "url": f"https://reddit.com{post.get('permalink')}" if post.get("permalink") else None,
            "platform": "Reddit",
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"]
        })
    return pd.DataFrame(posts)

# ----------------------------
# Main pipeline
# ----------------------------
if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    os.makedirs(output_dir, exist_ok=True)
    all_data = pd.DataFrame(columns=["domain","skill_query","title","description","source","publishedAt","url","platform","sentiment","sentiment_score"])
    total_queries = sum(len(skills) for skills in DOMAINS_AND_SKILLS.values())
    query_count = 0

    reddit_token = get_reddit_token()
    if not reddit_token:
        print("⚠️ Failed to get Reddit token. Skipping Reddit data.")

    for domain, skills in DOMAINS_AND_SKILLS.items():
        for skill in skills:
            query_count += 1
            print(f"\n[{query_count}/{total_queries}] Fetching for '{skill}' in '{domain}'...")

            df_newsapi = fetch_newsapi(skill, domain)
            all_data = pd.concat([all_data, df_newsapi], ignore_index=True)

            df_newsdata = fetch_newsdata(skill, domain)
            all_data = pd.concat([all_data, df_newsdata], ignore_index=True)

            df_googlenews = fetch_googlenews(skill, domain)
            all_data = pd.concat([all_data, df_googlenews], ignore_index=True)

            if reddit_token:
                df_reddit = fetch_reddit(skill, domain, reddit_token)
                all_data = pd.concat([all_data, df_reddit], ignore_index=True)

            time.sleep(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"{output_dir}/merged_with_sentiment_{timestamp}.csv"
    all_data.to_csv(output_file, index=False)
    print(f"\n✅ Data with sentiment saved to {output_file} ({len(all_data)} rows)")






