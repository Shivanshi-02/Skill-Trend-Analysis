import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from prophet import Prophet

class StrategicDashboard:
    def __init__(self):
        self.PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
        self.forecast_days = 5
        self.daily_sentiment = None
        self.load_data()

    def load_data(self):
        cleaned_files = list(self.PROCESSED_DIR.glob("cleaned_with_sentiment_*.csv"))
        if not cleaned_files:
            print("❌ No sentiment CSV found. Run sentiment pipeline first.")
            self.daily_sentiment = pd.DataFrame()
            return
        latest_file = max(cleaned_files, key=lambda f: f.stat().st_mtime)
        df = pd.read_csv(latest_file)
        if 'date' not in df.columns:
            if 'publishedAt' in df.columns:
                df.rename(columns={'publishedAt':'date'}, inplace=True)
            else:
                raise ValueError("CSV must contain 'date'")
        df['date'] = pd.to_datetime(df['date']).dt.date
        self.daily_sentiment = df.groupby(['date','skill'])['sentiment_score'].mean().reset_index()

    def forecast_skill(self, skill):
        df_skill = self.daily_sentiment[self.daily_sentiment['skill']==skill].copy()
        if df_skill.empty:
            return pd.DataFrame(), pd.DataFrame()
        prophet_df = df_skill.rename(columns={'date':'ds','sentiment_score':'y'})
        m = Prophet(daily_seasonality=True)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=self.forecast_days)
        forecast = m.predict(future)
        forecast_df = forecast[['ds','yhat']].tail(self.forecast_days).rename(columns={'ds':'date','yhat':'forecast_score'})
        return df_skill, forecast_df

    def detect_anomalies(self, df):
        if df.empty:
            return pd.DataFrame()
        mean = df['sentiment_score'].mean()
        std = df['sentiment_score'].std()
        return df[abs(df['sentiment_score']-mean) > 2*std]

    def plot_comparison_bar(self, user_skill):
        if self.daily_sentiment.empty:
            return None
        latest_date = self.daily_sentiment['date'].max()
        latest = self.daily_sentiment[self.daily_sentiment['date']==latest_date].copy()
        sorted_skills = latest.sort_values('sentiment_score', ascending=False)
        if user_skill not in sorted_skills['skill'].values:
            return None
        user_row = sorted_skills[sorted_skills['skill']==user_skill]
        top_4_others = sorted_skills[sorted_skills['skill']!=user_skill].head(4)
        top5 = pd.concat([user_row, top_4_others]).drop_duplicates(subset=['skill']).sort_values('sentiment_score', ascending=False)
        fig, ax = plt.subplots(figsize=(8,5))
        colors = ['#fb923c' if s==user_skill else '#3b82f6' for s in top5['skill']]
        ax.bar(top5['skill'], top5['sentiment_score'], color=colors, edgecolor='black')
        ax.set_ylim(0,1)
        ax.set_title(f"Sentiment Comparison: '{user_skill}' vs Others", fontsize=14, fontweight='bold')
        ax.set_ylabel("Sentiment Score")
        ax.set_xticklabels(top5['skill'], rotation=15)
        fig.tight_layout()
        return fig

    def plot_forecast_chart(self, user_skill):
        """Forecast chart: only forecasted next days, x-axis fixed to start after last historical date"""
        history_df, forecast_df = self.forecast_skill(user_skill)
        if forecast_df.empty:
            print(f"No forecast data for {user_skill}")
            return None

        # Get last historical date
        if not history_df.empty:
            last_date = max(history_df['date'])
        else:
            last_date = pd.Timestamp.today().date()

        # Generate forecast dates starting after last historical date
        forecast_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1),
                                       periods=len(forecast_df),
                                       freq='D')

        fig, ax = plt.subplots(figsize=(12,5))
        # Use generated dates for x-axis
        ax.plot(forecast_dates, forecast_df['forecast_score'].values, color='#fb923c',
                linestyle='--', marker='o', label="Forecast")

        ax.set_title(f"Forecast for {user_skill}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Sentiment Score")
        ax.set_xlabel("Date")
        ax.set_ylim(0,1)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_trend_chart(self, user_skill):
        """Trend chart: Actual + anomalies"""
        history_df, _ = self.forecast_skill(user_skill)
        if history_df.empty:
            print(f"No data for {user_skill}")
            return None

        anomalies_df = self.detect_anomalies(history_df)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(history_df['date'], history_df['sentiment_score'], color='#3b82f6', marker='o', label="Actual")
        if not anomalies_df.empty:
            ax.scatter(anomalies_df['date'], anomalies_df['sentiment_score'], color='red', s=100, label='Anomalies', edgecolor='black')
        ax.set_title(f"Trend for {user_skill}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Sentiment Score")
        ax.set_xlabel("Date")
        ax.set_ylim(0,1)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        return fig
