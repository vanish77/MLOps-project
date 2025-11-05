"""
Example of inference with trained model for review classification.

Usage:
    python example_inference.py
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def predict_sentiment(text: str, model_path: str = "./artefacts/distilbert-imdb"):
    """
    Predict text sentiment.
    
    Args:
        text: Review text
        model_path: Path to trained model
        
    Returns:
        Tuple[str, float]: (prediction, confidence)
    """
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = logits.argmax(-1).item()
        confidence = probs[0, prediction].item()
    
    sentiment = "Positive ?" if prediction == 1 else "Negative ?"
    return sentiment, confidence


if __name__ == "__main__":
    # Example reviews
    reviews = [
        "This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout.",
        "Waste of time. Boring storyline and terrible acting. Would not recommend.",
        "It was okay, nothing special. Some good moments but overall forgettable.",
        "One of the best films I've seen this year! Must watch!",
        "Disappointing. Had high expectations but the movie fell flat.",
    ]
    
    print("=" * 80)
    print("IMDb Sentiment Analysis - Inference Example")
    print("=" * 80)
    
    for i, review in enumerate(reviews, 1):
        sentiment, confidence = predict_sentiment(review)
        print(f"\nReview {i}:")
        print(f"  Text: {review}")
        print(f"  Prediction: {sentiment}")
        print(f"  Confidence: {confidence:.1%}")
    
    print("\n" + "=" * 80)
