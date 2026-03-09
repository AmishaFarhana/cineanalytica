"""
Movie Recommendation Script

Provides content-based, collaborative filtering (SVD), and hybrid recommendations.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path to import app modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.recommenders import (
    recommend_by_title,
    recommend_svd_for_user,
    hybrid_recommend_for_user,
    load_hybrid_params
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate movie recommendations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Recommendation mode arguments
    parser.add_argument('--title', type=str, default=None,
                       help='Movie title for content-based recommendations')
    parser.add_argument('--user_id', type=int, default=None,
                       help='User ID for collaborative filtering recommendations')
    parser.add_argument('--hybrid', action='store_true',
                       help='Use hybrid recommendations (requires --user_id)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of recommendations to return')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Hybrid alpha weight (default: from saved params)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.title and args.user_id:
        print("❌ Error: Specify either --title OR --user_id, not both")
        sys.exit(1)
    
    if not args.title and not args.user_id:
        print("❌ Error: Must specify either --title or --user_id")
        parser.print_help()
        sys.exit(1)
    
    if args.hybrid and not args.user_id:
        print("❌ Error: --hybrid requires --user_id")
        sys.exit(1)
    
    print("=" * 70)
    print("CineAnalytica Recommendation Engine")
    print("=" * 70)
    
    try:
        # Content-based recommendations
        if args.title:
            print(f"\nContent-Based Recommendations for: '{args.title}'")
            print("-" * 70)
            
            result = recommend_by_title(args.title, top_n=args.top_n)
            
            if isinstance(result, dict) and 'error' in result:
                print(f"\n❌ {result['error']}")
                if result.get('did_you_mean'):
                    print("\nDid you mean:")
                    for suggestion in result['did_you_mean'][:5]:
                        print(f"  - {suggestion}")
            else:
                print(f"\nTop {len(result)} Similar Movies:\n")
                for idx, row in result.iterrows():
                    print(f"  {idx+1}. {row['title']}")
                    print(f"     Similarity: {row['similarity_score']:.4f}\n")
        
        # SVD collaborative filtering
        elif args.user_id and not args.hybrid:
            print(f"\nCollaborative Filtering (SVD) for User {args.user_id}")
            print("-" * 70)
            
            result = recommend_svd_for_user(args.user_id, top_n=args.top_n)
            
            if result.empty:
                print(f"\n⚠️  No recommendations found for user {args.user_id}")
                print("User may not exist or has no rating history.")
            else:
                print(f"\nTop {len(result)} Predicted Movies:\n")
                for idx, row in result.iterrows():
                    print(f"  {idx+1}. {row['title']}")
                    print(f"     Predicted Rating: {row['pred_rating']:.2f}")
                    print(f"     Genres: {row['genre_text']}\n")
        
        # Hybrid recommendations
        else:  # args.user_id and args.hybrid
            # Load alpha from saved params if not specified
            if args.alpha is None:
                try:
                    params = load_hybrid_params()
                    alpha = params.get('alpha', 0.7)
                except Exception:
                    alpha = 0.7
                    print(f"⚠️  Could not load hybrid params, using default alpha={alpha}")
            else:
                alpha = args.alpha
            
            print(f"\nHybrid Recommendations for User {args.user_id}")
            print(f"Alpha (SVD weight): {alpha:.2f}")
            print("-" * 70)
            
            result = hybrid_recommend_for_user(
                args.user_id, alpha=alpha, top_n=args.top_n
            )
            
            if result.empty:
                print(f"\n⚠️  No recommendations found for user {args.user_id}")
                print("User may not exist or has insufficient rating history.")
            else:
                print(f"\nTop {len(result)} Hybrid Recommendations:\n")
                for idx, row in result.iterrows():
                    print(f"  {idx+1}. {row['title']}")
                    print(f"     Hybrid Score: {row['hybrid_score']:.4f}")
                    print(f"     Genres: {row['genre_text']}\n")
        
        print("=" * 70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure all required models and database files exist:")
        print("  - data/cineanalytica.db")
        print("  - models/svd_recommender.joblib")
        print("  - models/hybrid_params.joblib")
        sys.exit(1)


if __name__ == "__main__":
    main()
