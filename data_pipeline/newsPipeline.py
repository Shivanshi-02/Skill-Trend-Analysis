import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import requests
import time
from requests.auth import HTTPBasicAuth

# ----------------------------
# Load environment variables from .env
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

# ----------------------------
# Load domains & skills dynamically from JSON
# ----------------------------
with open("skills.json") as f:
    DOMAINS_AND_SKILLS = json.load(f)

# ----------------------------
# Fetch NewsAPI articles
# ----------------------------
def fetch_newsapi(query, domain, max_results=20):
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
        # Parse publishedAt safely
        published_at = pd.to_datetime(a.get("publishedAt"), errors='coerce')
        articles.append({
            "domain": domain,
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": published_at.isoformat() if published_at is not pd.NaT else None,
            "url": a.get("url"),
            "platform": "NewsAPI"
        })
    return pd.DataFrame(articles)

# ----------------------------
# Fetch NewsData.io articles
# ----------------------------
def fetch_newsdata(query, domain, max_results=20):
    params = {"apikey": NEWSDATA_TOKEN, "q": query, "language": "en"}
    try:
        response = requests.get(NEWSDATA_URL, params=params).json()
    except Exception as e:
        print(f"Error fetching NewsData.io for '{query}': {e}")
        return pd.DataFrame()
    
    articles = []
    for a in response.get("results", [])[:max_results]:
        # Parse pubDate safely
        published_at = pd.to_datetime(a.get("pubDate"), errors='coerce')
        articles.append({
            "domain": domain,
            "skill_query": query,
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source_id"),
            "publishedAt": published_at.isoformat() if published_at is not pd.NaT else None,
            "url": a.get("link"),
            "platform": "NewsData"
        })
    return pd.DataFrame(articles)

# ----------------------------
# Get Reddit 
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
        return pd.DataFrame()
    
    posts = []
    for p in response.get("data", {}).get("children", []):
        post = p["data"]
        # Parse created_utc safely
        created_at = datetime.fromtimestamp(post.get("created_utc")).isoformat() if post.get("created_utc") else None
        posts.append({
            "domain": domain,
            "skill_query": query,
            "title": post.get("title"),
            "description": post.get("selftext"),
            "source": post.get("subreddit"),
            "publishedAt": created_at,
            "url": f"https://reddit.com{post.get('permalink')}" if post.get("permalink") else None,
            "platform": "Reddit"
        })
    return pd.DataFrame(posts)

# ----------------------------
# Main pipeline
# ----------------------------
if __name__ == "__main__":
    output_dir = "../data/raw"
    os.makedirs(output_dir, exist_ok=True)
    all_data = pd.DataFrame()
    total_queries = sum(len(skills) for skills in DOMAINS_AND_SKILLS.values())
    query_count = 0

    # Get Reddit token once
    reddit_token = get_reddit_token()
    if not reddit_token:
        print("Failed to get Reddit token. Reddit data will be skipped.")

    # Loop through domains & skills dynamically
    for domain, skills in DOMAINS_AND_SKILLS.items():
        for skill in skills:
            query_count += 1
            print(f"[{query_count}/{total_queries}] Fetching news for '{skill}' in '{domain}'...")

            df_newsapi = fetch_newsapi(skill, domain)
            all_data = pd.concat([all_data, df_newsapi], ignore_index=True)
            time.sleep(1)

            df_newsdata = fetch_newsdata(skill, domain)
            all_data = pd.concat([all_data, df_newsdata], ignore_index=True)
            time.sleep(1)

            if reddit_token:
                df_reddit = fetch_reddit(skill, domain, reddit_token)
                all_data = pd.concat([all_data, df_reddit], ignore_index=True)
                time.sleep(1)

    # Save merged data to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"{output_dir}/merged_news_by_domain_{timestamp}.csv"
    all_data.to_csv(output_file, index=False)
    print(f"\n Data saved to {output_file} ({len(all_data)} rows)")






