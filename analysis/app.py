import streamlit as st
from pathlib import Path
import sys
import time

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from data_pipeline import newsPipeline, cleanPipeline
from analysis import sentimentPipeline
from analysis.dashboard import StrategicDashboard

# -----------------------------
st.set_page_config(
    page_title="Skill Trend Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Skill Trend Analyzer Dashboard")
st.markdown("Welcome! Enter a technology skill below to fetch news, clean data, and analyze trends.")

# -----------------------------
skill_input = st.text_input("Enter Technology Skill (e.g., Python, React, AI):", "").strip()

if skill_input:
    # --- Step 1: Fetch & Clean News
    if st.button("Fetch & Clean News"):
        st.text(f"🚀 Running full pipeline for '{skill_input}' ...")
        
        # 1️⃣ Fetch news for user + top 4 trending
        raw_csv_path = newsPipeline.fetch_news_for_user_skill(skill_input)
        st.text(f"✅ Raw data saved: {raw_csv_path}")

        # 2️⃣ Clean CSV
        cleanPipeline.run_clean_pipeline()
        st.text("✅ Cleaned tech CSV saved.")

    # --- Step 2: Analyze Sentiment & Show Trends
    if st.button("Analyze Sentiment & Show Trends"):
        st.text("📌 Running sentiment analysis on latest cleaned tech CSV...")
        sentiment_df = sentimentPipeline.run_sentiment_pipeline_on_latest()

        if sentiment_df.empty:
            st.warning("No sentiment data available. Make sure cleaned CSV exists.")
        else:
            # Filter for user skill
            available_skills = sentiment_df['skill'].unique()
            matched_skill = next((s for s in available_skills if s.lower() == skill_input.lower()), None)

            if not matched_skill:
                st.warning(f"Skill '{skill_input}' not found in dataset.")
            else:
                user_skill = matched_skill
                st.success(f"Skill found: **{user_skill}**")

                # -----------------------------
                # Display dashboard charts
                dashboard = StrategicDashboard()
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sentiment Comparison Bar Chart")
                    st.pyplot(dashboard.plot_comparison_bar(user_skill))

                with col2:
                    st.subheader("Forecast & Trend Chart")
                    st.pyplot(dashboard.plot_user_forecast(user_skill))

else:
    st.info("Please enter a technology skill to begin analysis.")

