# analysis/app.py
import streamlit as st
from pathlib import Path
import sys
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import plotly.express as px

# -----------------------------
# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Make sure .env is available to Streamlit process
load_dotenv()

# Import all necessary modules
from data_pipeline import newsPipeline, cleanPipeline
from analysis import sentimentPipeline
from analysis.dashboard import StrategicDashboard
from trend_forecasting.run_forecasting import run_forecasting_and_alerts, send_slack_alert
# -----------------------------

# -----------------------------
# Configuration and Initialization
# -----------------------------
st.set_page_config(
    page_title="Skills Trend Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for sentiment and skill
if "daily_sentiment" not in st.session_state:
    st.session_state["daily_sentiment"] = None
if "skill_input" not in st.session_state:
    st.session_state["skill_input"] = "Python"  # Default value

# -----------------------------
# Sidebar Content (Enhanced Download Section)
# -----------------------------
with st.sidebar:
      # --- Enhanced 'App Configurations' Header ---
    st.markdown("""
        <h2 style='
            border: 1px solid #383942;
            padding-bottom: 5px; 
            margin-top: 0;
            font-weight: 800; /* Extra bold */
        '>
            ⚙️ App Configurations
        </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Last Updated Info (Dynamic Timestamp)
    st.subheader("🕒 Last Analysis Time")
    # Using a style that fits dark mode better
    st.markdown(f"""
        <div style=' background-color: #e0fae8;color: #1f7a3f; padding: 10px; border-radius: 5px; border: 1px solid #383942;'>
            Updated at: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong>
        </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("---")

    # Alert Configuration Status
    st.subheader("📢 Alert Integration Status")
    st.markdown("""
        <div style='background-color: #e0fae8; padding: 10px; border-radius: 5px; color: #1f7a3f;'>
            <strong>🔔 Slack Alerts: Active</strong><br>
            <small>Forecast results will be sent.</small>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- Download latest CSVs  ----------
    st.subheader("💾 Download Data")
    
    # Use an expander for a cleaner look
    with st.expander("Available Datasets", expanded=True):
        try:
            # Latest raw CSV
            raw_dir = BASE_DIR / "data" / "raw"
            raw_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []
            latest_raw = max(raw_files, key=lambda f: f.stat().st_mtime) if raw_files else None

            if latest_raw:
                raw_bytes = latest_raw.read_bytes()
                st.download_button(
                    label=f"⬇️ Raw CSV File",
                    data=raw_bytes,
                    file_name=latest_raw.name,
                    mime="text/csv",
                    key='download_raw',
                    use_container_width=True
                )
            else:
                st.markdown("Raw CSV: **N/A**")


            # Latest cleaned_with_sentiment CSV
            processed_dir = BASE_DIR / "data" / "processed"
            cleaned_files = list(processed_dir.glob("cleaned_with_sentiment_*.csv")) if processed_dir.exists() else []
            latest_cleaned = max(cleaned_files, key=lambda f: f.stat().st_mtime) if cleaned_files else None

            if latest_cleaned:
                cleaned_bytes = latest_cleaned.read_bytes()
                st.download_button(
                    label=f"⬇️ Cleaned CSV File",
                    data=cleaned_bytes,
                    file_name=latest_cleaned.name,
                    mime="text/csv",
                    key='download_cleaned',
                    use_container_width=True
                )
            else:
                st.markdown("Cleaned CSV: **N/A**")
                
        except Exception as e:
            st.error(f"❌ Download error: {e}")


# -----------------------------
# Main App Content
# -----------------------------
st.title("🚀 Skills Trend Analyzer")
st.markdown("Welcome! Analyze any technology skill and learn about latest trends...")

# Skill Input remains central
st.session_state["skill_input"] = st.text_input(
    "Discover the Future Trajectory of Skill (e.g., Python, AI):",
    value=st.session_state["skill_input"]
).strip()
skill_input = st.session_state["skill_input"]

# -----------------------------
# Workflow Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Forecast & Alerts",
    "Charts"
])

# -----------------------------
# TAB 1: Overall Analytics (Initial Data Status) 
# -----------------------------
with tab1:
    st.header("📊 Current Data Overview")
    st.info("This tab shows the overall metrics and sentiment distribution of the latest available cleaned dataset.")

    # ---------- CLEANER BUTTON LAYOUT (Fetch/Clean/Sentiment) ----------
    st.markdown("#### ⚙️ Generate Analysis Reports")
    col_fetch, col_sentiment = st.columns(2)

    with col_fetch:
        # Button 1: Fetch & Clean
        if st.button("Start Analysis", use_container_width=True):
            if not skill_input:
                st.warning("Please enter a skill first.")
            else:
                with st.spinner(f"🚀 Running full pipeline for '{skill_input}' ..."):
                    try:
                        # 1️⃣ Fetch news
                        raw_csv_path = newsPipeline.fetch_news_for_user_skill(skill_input)
                        # 2️⃣ Clean CSV
                        cleanPipeline.run_clean_pipeline()
                        st.success(f"✅ Data fetched and cleaned. Ready for sentiment analysis.")
                        st.session_state["last_run_success"] = True # Set a success flag
                    except Exception as e:
                        st.error(f"❌ Pipeline error: {e}")
                        st.session_state["last_run_success"] = False

    with col_sentiment:
        # Button 2: Run Sentiment Analysis (Refresh Data Status)
        if st.button("Analyse Sentiments", type="primary", use_container_width=True):
            with st.spinner("📌 Running sentiment analysis ..."):
                try:
                    # Reruns sentiment on the latest data file
                    st.session_state['daily_sentiment'] = sentimentPipeline.run_sentiment_pipeline_on_latest()
                    st.success("✅ Sentiment analysis completed and data status refreshed!")
                except Exception as e:
                    st.error(f"❌ Sentiment pipeline exception: {e}")
                    st.session_state['daily_sentiment'] = None
    
    st.markdown("---") # Separator before metrics

    # ---------- DATASET DETAILS & CHARTS (Rest of Tab 1) ----------
    if st.session_state['daily_sentiment'] is not None:
        st.subheader("Dataset Details")
        st.write(f"Data includes analysis for **{len(st.session_state['daily_sentiment']['skill'].unique())}** unique skills.")

        # Re-using the Overall Analytics logic
        try:
            processed_dir = BASE_DIR / "data" / "processed"
            cleaned_files = list(processed_dir.glob("cleaned_with_sentiment_*.csv"))

            if cleaned_files:
                latest_cleaned = max(cleaned_files, key=lambda f: f.stat().st_mtime)
                df_cleaned = pd.read_csv(latest_cleaned)

                # --- METRICS ---
                total_articles = len(df_cleaned)
                sent_col = next((c for c in ['sentiment', 'sentiment_label', 'label', 'sentiment_text'] if c in df_cleaned.columns), None)

                col1, col2, col3 = st.columns(3)

                if sent_col is not None:
                    sent_series = df_cleaned[sent_col].astype(str).str.lower()
                    pos_pct = round(100 * (sent_series == 'positive').mean(), 1)
                    # Handle missing sentiment_score column gracefully for avg score
                    avg_sent_score = round(float(df_cleaned.get('sentiment_score', pd.Series([0])).mean()), 3) if 'sentiment_score' in df_cleaned.columns else "N/A"

                    with col1:
                        st.metric(label="Positive %", value=f"{pos_pct}%")
                    with col2:
                        st.metric(label="Avg Sentiment Score", value=f"{avg_sent_score}")
                    with col3:
                        st.metric(label="Total Articles", value=f"{total_articles}")

                    st.markdown("---")

                    # --- PIE CHART ---
                    sentiment_counts = sent_series.value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].str.capitalize()
                    color_map = {'Positive': '#4ef39c', 'Negative': '#f54b86', 'Neutral': '#8aecff'}

                    pie_fig = px.pie(
                        sentiment_counts, names='Sentiment', values='Count', color='Sentiment',
                        color_discrete_map=color_map, title="Sentiment Distribution in Latest Dataset"
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                    # --- BAR CHART: Day-wise Article Count Trend (Kept Fixed Y-Axis as per last code block) ---
                    if 'date' in df_cleaned.columns:
                        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
                        daily_counts = df_cleaned.groupby(df_cleaned['date'].dt.date).size().reset_index(name='Articles')
                        daily_counts['date_str'] = daily_counts['date'].astype(str)

                        bar_fig = px.bar(
                            daily_counts, x='date_str', y='Articles', text='Articles',
                            title="🗓️ Interactive Article Count Trend",
                            color='Articles',
                            color_continuous_scale=[(0, '#00A389'), (1, '#00A389')]
                        )
                        bar_fig.update_traces(textposition='outside')
                        bar_fig.update_layout(
                            template="plotly_dark",
                            title_text="Interactive Article Count Trend",
                            xaxis_title="",
                            xaxis_tickangle=-30,
                            yaxis=dict(
                                title="Article_Count",
                                dtick=2, range=[0, 14], tick0=0,
                                gridcolor='rgba(200,200,200,0.1)',
                                rangemode='tozero'
                            ),
                            coloraxis_showscale=False,
                            bargap=0.05
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)
                    else:
                        st.info("`date` column not found for trend chart.")

                else:
                    st.warning("Sentiment label column not found for metrics/pie chart.")
            else:
                st.warning("No cleaned_with_sentiment CSV files found in /data/processed. Run data pipelines.")
        except Exception as e:
            st.error(f"❌ Error loading or generating overall charts: {e}")
    else:
        st.info("Click 'Analyse Sentiments' above to load more details.")

# -----------------------------
# TAB 2: Forecast & Alerts (No change from last version)
# -----------------------------
with tab2:
    st.header("🔔 Trend Forecasting & Alerts")
    st.info("Click below to see significant trend shifts...")

    if st.session_state['daily_sentiment'] is None:
        st.warning("⚠️  Please ensure running sentiment analysis before proceeding!")
    else:
        if st.button("Start Forecasting & Send Alerts", type="primary"):
            with st.spinner("Processing forecast and triggering Slack alerts..."):
                try:
                    alert_results = run_forecasting_and_alerts()
                    st.success("✅ Forecasting completed and alerts processed.")
                except Exception as e:
                    st.error(f"❌ run_forecasting_and_alerts() raised an exception: {e}")
                    alert_results = []

            st.subheader("Skills-wise Alerts")
            if alert_results:
                st.dataframe(pd.DataFrame(alert_results))
            else:
                st.info("No significant trend alerts were triggered.")

# -----------------------------
# TAB 3: Skill-Specific Charts (No change from last version)
# -----------------------------
with tab3:
    st.header(f"📈 Strategic Charts for: **{skill_input}**")

    if st.session_state['daily_sentiment'] is None:
        st.warning("⚠️ Please ensure running sentiment analysis before proceeding!")
    else:
        daily_sentiment = st.session_state['daily_sentiment']

        # Match user skill
        matched_skill = next((s for s in daily_sentiment['skill'].unique() if s.lower() == skill_input.lower()), None)

        if not matched_skill:
            st.error(f"Skill '**{skill_input}**' not found in the dataset. Try one of the available skills shown in Tab 1.")
        else:
            user_skill = matched_skill
            st.success(f"Loading charts for: **{user_skill}**")

            # Initialize Dashboard (will load data internally based on latest CSV)
            dashboard = StrategicDashboard()

            # Chart 1: Trend Chart (Actual + Anomalies)
            st.subheader("1. Historical Sentiment Trend Analysis")
            fig_trend = dashboard.plot_trend_chart(user_skill)
            if fig_trend:
                st.pyplot(fig_trend)
            else:
                st.warning("Not enough data for the historical trend chart.")

            # Chart 2: Forecast Chart
            st.subheader("2. 5-Days Sentiment Forecasting")
            fig_forecast = dashboard.plot_forecast_chart(user_skill)
            if fig_forecast:
                st.pyplot(fig_forecast)
            else:
                st.warning("Not enough data for the forecast chart.")

            # Chart 3: Sentiment Comparison Bar Chart
            st.subheader("3. Skill Comparison with its Competitors")
            fig1 = dashboard.plot_comparison_bar(user_skill)
            if fig1:
                st.pyplot(fig1)
            else:
                st.warning("Not enough data for the comparison chart.")


                    




                    