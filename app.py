import time
import sqlite3
from datetime import datetime

import feedparser
import pandas as pd
import pytz
import streamlit as st
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, util
from streamlit_autorefresh import st_autorefresh

# =====================================================
# CONFIG
# =====================================================

RSS_URL = "https://www.reddit.com/r/AskReddit/new/.rss"
DB_NAME = "reddit_posts.db"
REFRESH_SECONDS = 200

POLAND_TZ = pytz.timezone("Europe/Warsaw")

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(page_title="AskReddit Monitor", layout="wide")
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="auto_refresh")

# =====================================================
# SESSION STATE
# =====================================================

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# CATEGORY SYSTEM (IMPROVED)
# =====================================================

CATEGORY_SEEDS = {
    "Relationships": "romantic relationships dating marriage breakup cheating love conflict",
    "Confessions": "secret guilt hidden truth personal confession regret shame",
    "Psychology": "human behavior thoughts emotions decision making mindset",
    "Social Issues": "society discrimination inequality social problems politics culture",
    "Ethics": "moral dilemma right wrong ethical decision philosophy",
    "Money": "salary debt finance wealth financial struggle money problems",
    "Career": "job workplace promotion career choice professional experience",
    "Nostalgia": "childhood memories past school old times remembering history",
    "Controversial": "hot take debate argument polarizing opinion disagreement",
    "Funny": "humor joke embarrassing funny story laughter awkward situation",
    "Hypothetical": "what if imaginary scenario thought experiment possibility",
    "Fear": "scary experience danger anxiety horror fear trauma",
    "Family": "parents siblings household family conflict upbringing",
    "Dating": "dating apps crush attraction romance relationship questions",
    "Technology": "phones apps internet AI tech problems digital world",
    "Society": "culture norms behavior trends society discussion",
    "Life Advice": "advice guidance decision help suggestion life choice",
    "Human Behavior": "why people act psychology behavior patterns social reactions",
    "General Discussion": "open question broad topic general opinion discussion"
}

CATEGORIES = list(CATEGORY_SEEDS.keys())
category_embeddings = model.encode(list(CATEGORY_SEEDS.values()), convert_to_tensor=True)

# =====================================================
# ENGAGEMENT ARCHETYPES (NEW CORE LOGIC)
# =====================================================

ENGAGEMENT_ARCHETYPES = [
    "high emotional personal confession people relate to",
    "controversial moral or ethical dilemma",
    "embarrassing or funny real life story",
    "hypothetical imagination question what if scenario",
    "relationship or dating conflict situation",
    "extreme unusual or shocking experience",
    "nostalgic childhood or past memory question",
    "money career or life decision stress question",
    "weird or unexpected situation inviting stories",
    "broad curiosity question inviting opinions"
]

archetype_embeddings = model.encode(ENGAGEMENT_ARCHETYPES, convert_to_tensor=True)

# =====================================================
# DATABASE
# =====================================================

conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    posted_time TEXT,
    engagement_score INTEGER,
    reason TEXT,
    category TEXT,
    fetched_at TEXT
)
""")
conn.commit()

# =====================================================
# HELPERS
# =====================================================

def format_time(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = dt.astimezone(POLAND_TZ)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return ts

def post_exists(post_id):
    cursor.execute("SELECT 1 FROM posts WHERE post_id=?", (post_id,))
    return cursor.fetchone() is not None

def save_post(post_id, title, url, posted_time, score, reason, category):
    cursor.execute("""
        INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        post_id, title, url, posted_time,
        score, reason, category,
        datetime.utcnow().isoformat()
    ))
    conn.commit()

# =====================================================
# CATEGORY CLASSIFICATION (IMPROVED)
# =====================================================

def get_category(title):
    emb = model.encode(title, convert_to_tensor=True)
    sims = util.cos_sim(emb, category_embeddings)[0]
    return CATEGORIES[int(np.argmax(sims))]

# =====================================================
# ENGAGEMENT SCORING (REPLACED LOGIC)
# =====================================================

def engagement_multiplier(title):
    t = title.lower()
    m = 1.0

    if t.startswith("what"):
        m += 0.10
    if "would you" in t:
        m += 0.20
    if "have you ever" in t:
        m += 0.25
    if t.count("?") >= 1:
        m += 0.10

    return m

def score_title(title):
    emb = model.encode(title, convert_to_tensor=True)

    sims = util.cos_sim(emb, archetype_embeddings)[0]
    base = float(torch.max(sims))

    score = base * 100

    # lightweight boosts
    t = title.lower()

    if any(w in t for w in ["secret", "regret", "worst", "embarrassed"]):
        score += 8

    if len(title.split()) > 15:
        score += 5

    if len(title.split()) < 6:
        score -= 5

    score = score * engagement_multiplier(title)

    return int(max(1, min(100, score)))

# =====================================================
# REASON ENGINE (IMPROVED)
# =====================================================

def generate_reason(title):
    t = title.lower()
    reasons = []

    if "would you" in t:
        reasons.append("hypothetical engagement hook")
    if "have you ever" in t:
        reasons.append("personal experience trigger")
    if "what if" in t:
        reasons.append("imaginative scenario hook")
    if any(w in t for w in ["secret", "regret", "embarrassed", "worst"]):
        reasons.append("strong emotional trigger")
    if "?" in t:
        reasons.append("question format increases comments")
    if len(title.split()) > 15:
        reasons.append("story-like structure likely to generate replies")

    if not reasons:
        reasons.append("neutral but broad appeal")

    return " | ".join(reasons)

# =====================================================
# ANALYSIS
# =====================================================

def analyze_titles(titles):
    results = []
    for t in titles:
        cat = get_category(t)
        sc = score_title(t)
        reason = generate_reason(t)

        results.append({
            "score": sc,
            "category": cat,
            "reason": reason
        })
    return results

# =====================================================
# FETCH POSTS
# =====================================================

def fetch_posts():
    feed = feedparser.parse(RSS_URL)

    new_posts = []

    for e in feed.entries:
        try:
            url = e.link
            post_id = url.split("/comments/")[1].split("/")[0]

            if post_exists(post_id):
                continue

            new_posts.append({
                "id": post_id,
                "title": e.title,
                "url": url,
                "time": e.published
            })
        except:
            continue

    BATCH = 5

    for i in range(0, len(new_posts), BATCH):
        batch = new_posts[i:i+BATCH]
        titles = [x["title"] for x in batch]

        results = analyze_titles(titles)

        for item, res in zip(batch, results):
            save_post(
                item["id"],
                item["title"],
                item["url"],
                item["time"],
                res["score"],
                res["reason"],
                res["category"]
            )

# =====================================================
# INIT
# =====================================================

cursor.execute("SELECT COUNT(*) FROM posts")
if cursor.fetchone()[0] == 0:
    fetch_posts()

if time.time() - st.session_state.last_refresh > REFRESH_SECONDS:
    fetch_posts()
    st.session_state.last_refresh = time.time()

# =====================================================
# UI
# =====================================================

df = pd.read_sql_query("""
SELECT title, url, posted_time,
       engagement_score, reason, category, fetched_at
FROM posts
ORDER BY datetime(fetched_at) DESC
""", conn)

df.columns = ["Title","Reddit Link","Posted","Score","Reason","Category","Fetched At"]

df["Posted"] = df["Posted"].apply(format_time)
df["Fetched At"] = df["Fetched At"].apply(format_time)

st.title("🔥 AskReddit Engagement Monitor")

last_ref = datetime.fromtimestamp(
    st.session_state.last_refresh
).astimezone(POLAND_TZ).strftime("%d/%m/%Y %H:%M:%S")

st.markdown(f"### ⏱ Last refreshed: **{last_ref}**")

if st.button("🔄 Refresh Now"):
    fetch_posts()
    st.session_state.last_refresh = time.time()
    st.rerun()

st.subheader("📋 Latest Posts")
st.dataframe(df, use_container_width=True)

st.subheader("🚀 Top Posts")
st.dataframe(df.sort_values("Score", ascending=False).head(10),
             use_container_width=True)

st.subheader("📊 Stats")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Posts", len(df))

with c2:
    st.metric("Avg Score",
              round(df["Score"].mean(), 1) if len(df) else 0)

with c3:
    st.metric("Max Score",
              df["Score"].max() if len(df) else 0)

st.caption("Auto-refresh every 200 seconds | Improved embedding-based scoring")