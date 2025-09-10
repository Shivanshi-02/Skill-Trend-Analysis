from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load a general sentiment model 
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
      # Handle edge case: if the text is empty or just spaces
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    try:
         # Run the sentiment pipeline
        result = sentiment_pipeline(
            text,
            truncation=True,
            max_length=512
        )[0]

        # Mapping labels of our model
        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }

        # Convert raw label into mapped label
        label = label_map.get(result["label"], result["label"])

        # Return dictionary with readable label and rounded confidence score
        return {"label": label, "score": round(result["score"], 4)}

#Return safe default NEUTRAL if anything goes wrong
    except Exception as e:
        print("⚠️ Sentiment analysis failed:", e)
        return {"label": "NEUTRAL", "score": 0.0}



