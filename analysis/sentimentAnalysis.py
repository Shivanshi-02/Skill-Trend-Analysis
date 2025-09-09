from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load a general sentiment model (better for tech/news/social text)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    """
    Run sentiment analysis on given text using CardiffNLP RoBERTa.
    Handles empty or long input safely.
    Returns: {label, score}
    """
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    try:
        result = sentiment_pipeline(
            text,
            truncation=True,
            max_length=512
        )[0]

        # Normalize labels (CardiffNLP returns: "LABEL_0", "LABEL_1", "LABEL_2")
        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }
        label = label_map.get(result["label"], result["label"])

        return {"label": label, "score": round(result["score"], 4)}

    except Exception as e:
        print("⚠️ Sentiment analysis failed:", e)
        return {"label": "NEUTRAL", "score": 0.0}


#  Example run
# if __name__ == "__main__":
#     sample = "PyTorch adoption is growing rapidly and developers love it!"
#     print(analyze_sentiment(sample))


