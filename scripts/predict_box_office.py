"""
Box Office Revenue Prediction Script

Loads the trained XGBoost model and predicts box office revenue based on movie features.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "box_office_xgb.joblib"


def predict_box_office(budget, runtime, popularity, vote_average, vote_count,
                       release_year, release_month, log_star_power=None):
    """
    Predict box office revenue using the trained XGBoost model.
    
    Args:
        budget (float): Movie budget in dollars
        runtime (float): Runtime in minutes
        popularity (float): Popularity score
        vote_average (float): Average vote rating
        vote_count (int): Number of votes
        release_year (int): Release year
        release_month (int): Release month (1-12)
        log_star_power (float, optional): Log-transformed star power metric
    
    Returns:
        tuple: (log_revenue, revenue)
    """
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please ensure the box office model has been trained and saved.")
        sys.exit(1)
    
    # Load model
    model = joblib.load(MODEL_PATH)
    
    # Build feature DataFrame
    # Note: Adjust column names based on actual model training
    features = {
        'budget': [budget],
        'runtime': [runtime],
        'popularity': [popularity],
        'vote_average': [vote_average],
        'vote_count': [vote_count],
        'release_year': [release_year],
        'release_month': [release_month]
    }
    
    if log_star_power is not None:
        features['log_star_power'] = [log_star_power]
    
    X = pd.DataFrame(features)
    
    # Predict log revenue
    log_revenue_pred = model.predict(X)[0]
    
    # Convert back to actual revenue
    revenue_pred = np.expm1(log_revenue_pred)
    
    return log_revenue_pred, revenue_pred


def main():
    parser = argparse.ArgumentParser(
        description='Predict box office revenue for a movie',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--budget', type=float, required=True,
                       help='Movie budget in dollars')
    parser.add_argument('--runtime', type=float, required=True,
                       help='Runtime in minutes')
    parser.add_argument('--popularity', type=float, required=True,
                       help='Popularity score')
    parser.add_argument('--vote_average', type=float, required=True,
                       help='Average vote rating (0-10)')
    parser.add_argument('--vote_count', type=int, required=True,
                       help='Number of votes')
    parser.add_argument('--release_year', type=int, required=True,
                       help='Release year')
    parser.add_argument('--release_month', type=int, required=True,
                       help='Release month (1-12)')
    parser.add_argument('--log_star_power', type=float, default=None,
                       help='Log-transformed star power metric (optional)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Box Office Revenue Prediction")
    print("=" * 70)
    print("\nInput Features:")
    print(f"  Budget:         ${args.budget:,.0f}")
    print(f"  Runtime:        {args.runtime} minutes")
    print(f"  Popularity:     {args.popularity}")
    print(f"  Vote Average:   {args.vote_average}")
    print(f"  Vote Count:     {args.vote_count:,}")
    print(f"  Release:        {args.release_year}-{args.release_month:02d}")
    if args.log_star_power:
        print(f"  Log Star Power: {args.log_star_power}")
    
    # Make prediction
    log_revenue, revenue = predict_box_office(
        args.budget, args.runtime, args.popularity, args.vote_average,
        args.vote_count, args.release_year, args.release_month,
        args.log_star_power
    )
    
    print("\n" + "-" * 70)
    print("Prediction Results:")
    print(f"  Log Revenue:    {log_revenue:.4f}")
    print(f"  Revenue:        ${revenue:,.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
