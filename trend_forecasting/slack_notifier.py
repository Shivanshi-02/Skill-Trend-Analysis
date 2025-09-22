"""
    Sends each alert message to a Slack channel using an Incoming Webhook.
    """
import os
from dotenv import load_dotenv
import requests

load_dotenv()  # make sure this is here!

def send_slack_alert(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("SLACK_WEBHOOK_URL not found in .env")
    requests.post(webhook_url, json={"text": message})
