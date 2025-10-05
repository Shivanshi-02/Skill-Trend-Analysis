# analysis/dashboard.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from prophet import Prophet
from analysis.sentimentPipeline import run_sentiment_pipeline

class StrategicDashboard:
    def __init__(self):
        self.PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
        self.forecast_days = 5
        self.daily_sentiment = None
        self.load_data()

    def load_data(self):
        """Load latest sentiment CSV into daily aggregated sentiment."""
        cleaned_files = list(self.PROCESSED_DIR.glob("cleaned_with_sentiment_*.csv"))
        if not cleaned_files:
            print("❌ No sentiment CSV found. Run sentiment pipeline first.")
            self.daily_sentiment = pd.DataFrame()
            return
        latest_file = max(cleaned_files, key=lambda f: f.stat().st_mtime)
        df = pd.read_csv(latest_file)
        if 'date' not in df.columns:
            if 'publishedAt' in df.columns:
                df.rename(columns={'publishedAt': 'date'}, inplace=True)
            else:
                raise ValueError("CSV must contain 'date' column")
        df['date'] = pd.to_datetime(df['date']).dt.date
        # Daily average sentiment per skill
        self.daily_sentiment = df.groupby(['date', 'skill'])['sentiment_score'].mean().reset_index()

    def forecast_skill(self, skill):
        """Forecast sentiment for a skill using Prophet."""
        df_skill = self.daily_sentiment[self.daily_sentiment['skill'] == skill].copy()
        if df_skill.empty:
            return pd.DataFrame(), pd.DataFrame()
        prophet_df = df_skill.rename(columns={'date': 'ds', 'sentiment_score': 'y'})
        m = Prophet(daily_seasonality=True)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=self.forecast_days)
        forecast = m.predict(future)
        forecast_df = forecast[['ds', 'yhat']].tail(self.forecast_days).rename(columns={'ds': 'date', 'yhat': 'forecast_score'})
        return df_skill, forecast_df

    def detect_anomalies(self, df):
        """Detect anomalies: points >2 std deviations from mean."""
        if df.empty:
            return pd.DataFrame()
        mean = df['sentiment_score'].mean()
        std = df['sentiment_score'].std()
        anomalies = df[abs(df['sentiment_score'] - mean) > 2 * std].copy()
        return anomalies

    def plot_comparison_bar(self, user_skill):
        """Bar chart comparing user skill vs top 4 other skills."""
        if self.daily_sentiment.empty:
            return None
        latest_date = self.daily_sentiment['date'].max()
        latest = self.daily_sentiment[self.daily_sentiment['date'] == latest_date].copy()
        sorted_skills = latest.sort_values('sentiment_score', ascending=False)
        if user_skill not in sorted_skills['skill'].values:
            return None
        user_row = sorted_skills[sorted_skills['skill'] == user_skill]
        top_4_others = sorted_skills[sorted_skills['skill'] != user_skill].head(4)
        top5 = pd.concat([user_row, top_4_others]).drop_duplicates(subset=['skill'])
        top5 = top5.sort_values('sentiment_score', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#fb923c' if s == user_skill else '#3b82f6' for s in top5['skill']]
        ax.bar(top5['skill'], top5['sentiment_score'], color=colors, edgecolor='black')
        ax.set_ylim(0, 1)
        ax.set_title(f"Sentiment Comparison: '{user_skill}' vs Others", fontsize=14, fontweight='bold')
        ax.set_ylabel("Sentiment Score")
        ax.set_xticklabels(top5['skill'], rotation=15)
        fig.tight_layout()
        return fig

    def plot_user_forecast(self, user_skill):
        """Line chart: user skill trend + forecast."""
        history_df, forecast_df = self.forecast_skill(user_skill)
        if history_df.empty:
            print(f"No data found for skill {user_skill}")
            return None
        anomalies_df = self.detect_anomalies(history_df)

        fig, ax = plt.subplots(figsize=(12, 5))
        # Actual
        ax.plot(history_df['date'], history_df['sentiment_score'], color='#3b82f6', marker='o', label="Actual")
        # Forecast
        if not forecast_df.empty:
            ax.plot(forecast_df['date'], forecast_df['forecast_score'], color='#fb923c', linestyle='--', marker='o', label="Forecast")
        # Anomalies
        if not anomalies_df.empty:
            ax.scatter(anomalies_df['date'], anomalies_df['sentiment_score'], color='red', s=100, label='Anomalies', edgecolor='black')

        ax.set_title(f"Sentiment Trend & Forecast for {user_skill}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Sentiment Score")
        ax.set_xlabel("Date")
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        return fig


