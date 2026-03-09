"""
Sentiment Analysis Prediction Script

Loads the trained sentiment model and predicts sentiment for movie reviews.
"""

import argparse
from pathlib import Path
import joblib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"


def predict_sentiment(text):
    """
    Predict sentiment for a given text review.
    
    Args:
        text (str): Review text to analyze
    
    Returns:
        tuple: (label, probability) or (label, None) if probabilities unavailable
    """
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please ensure the sentiment model has been trained and saved.")
        sys.exit(1)
    
    # Load model (should be a pipeline with vectorizer + classifier)
    model = joblib.load(MODEL_PATH)
    
    # Predict
    prediction = model.predict([text])[0]
    
    # Try to get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba([text])[0]
            probability = max(proba)
        except Exception:
            pass
    
    return prediction, probability


def main():
    parser = argparse.ArgumentParser(
        description='Predict sentiment for movie review text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--text', type=str, required=True,
                       help='Review text to analyze')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Sentiment Analysis Prediction")
    print("=" * 70)
    print(f"\nReview Text:\n  \"{args.text}\"\n")
    
    # Make prediction
    label, probability = predict_sentiment(args.text)
    
    print("-" * 70)
    print("Prediction Results:")
    print(f"  Sentiment:  {label}")
    if probability is not None:
        print(f"  Confidence: {probability:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
