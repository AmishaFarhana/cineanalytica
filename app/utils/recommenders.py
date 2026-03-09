"""
Recommendation Engine Utilities for CineAnalytica

This module provides content-based, collaborative filtering (SVD), and hybrid
recommendation functions. When used in Streamlit, the build_content_recommender()
function is automatically cached with @st.cache_resource to avoid recomputing
TF-IDF and cosine similarity matrices on each run.

For other functions that load data, consider using @st.cache_data in your Streamlit
pages as appropriate.
"""

from pathlib import Path
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from functools import lru_cache

# Try to import streamlit for caching decorator
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "cineanalytica.db"
MODEL_DIR = PROJECT_ROOT / "models"


def load_svd_artifact():
    """
    Load the SVD recommender artifact from models/svd_recommender.joblib.
    
    Returns:
        dict: Contains keys 'svd' (trained SVD model) and 'movies_lookup' (DataFrame).
    """
    artifact_path = MODEL_DIR / "svd_recommender.joblib"
    return joblib.load(artifact_path)


def load_hybrid_params():
    """
    Load hybrid recommender parameters from models/hybrid_params.joblib.
    
    Returns:
        dict: Hybrid model parameters.
    """
    params_path = MODEL_DIR / "hybrid_params.joblib"
    return joblib.load(params_path)


def load_movies_from_db():
    """
    Load movies with genres from the SQLite database.
    
    Reads movie_id, title, overview from movies table and joins with
    movie_genres to aggregate genre names. Constructs combined_text
    for content-based filtering.
    
    Returns:
        pd.DataFrame: Columns [movie_id, title, overview, genre_text, combined_text].
    """
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        m.movie_id,
        m.title,
        m.overview,
        GROUP_CONCAT(g.genre_name, ' ') AS genre_text
    FROM movies m
    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
    LEFT JOIN genres g ON mg.genre_id = g.genre_id
    GROUP BY m.movie_id, m.title, m.overview
    """
    
    movies2 = pd.read_sql_query(query, conn)
    conn.close()
    
    # Fill missing values
    movies2['overview'] = movies2['overview'].fillna('')
    movies2['genre_text'] = movies2['genre_text'].fillna('')
    
    # Create combined text for content-based filtering
    movies2['combined_text'] = (
        movies2['title'] + ' ' + 
        movies2['overview'] + ' ' + 
        movies2['genre_text']
    )
    
    return movies2


def _build_content_recommender_impl():
    """
    Internal implementation of content recommender builder.
    Builds TF-IDF matrix and cosine similarity for content-based recommendations.
    
    Returns:
        dict: Contains 'movies2', 'tfidf', 'cosine_sim', 'title_to_idx', 'movieid_to_idx'.
    """
    movies2 = load_movies_from_db()
    
    # Build TF-IDF matrix
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=50000
    )
    tfidf_matrix = tfidf.fit_transform(movies2["genre_text"].fillna(""))
    
    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Build lookup indices
    title_to_idx = {title.lower(): idx for idx, title in enumerate(movies2['title'])}
    movieid_to_idx = {mid: idx for idx, mid in enumerate(movies2['movie_id'])}
    
    return {
        'movies2': movies2,
        'tfidf': tfidf,
        'cosine_sim': cosine_sim,
        'title_to_idx': title_to_idx,
        'movieid_to_idx': movieid_to_idx
    }


# Apply appropriate caching decorator based on environment
if HAS_STREAMLIT:
    build_content_recommender = st.cache_resource(_build_content_recommender_impl)
else:
    build_content_recommender = lru_cache(maxsize=1)(_build_content_recommender_impl)


def recommend_by_title(title, top_n=10):
    """
    Recommend movies similar to the given title using content-based filtering.
    
    Args:
        title (str): Movie title to find similar movies for.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame or dict: DataFrame with columns [title, similarity_score] if found,
                              or dict with 'error' and 'did_you_mean' if not found.
    """
    content_data = build_content_recommender()
    movies2 = content_data['movies2']
    cosine_sim = content_data['cosine_sim']
    title_to_idx = content_data['title_to_idx']
    
    title_lower = title.lower()
    
    # Check if title exists
    if title_lower not in title_to_idx:
        # Find close matches
        matches = [t for t in title_to_idx.keys() if title_lower in t][:5]
        return {
            "error": f"Title '{title}' not found in database",
            "did_you_mean": matches
        }
    
    # Get movie index and similarity scores
    idx = title_to_idx[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity (excluding the movie itself at position idx)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filter out the query movie itself and get top_n
    query_movie_id = movies2.iloc[idx]['movie_id']
    filtered_scores = []
    for i, score in sim_scores:
        # Skip if it's the same movie_id (handles duplicates)
        if movies2.iloc[i]['movie_id'] != query_movie_id:
            filtered_scores.append((i, score))
            if len(filtered_scores) >= top_n:
                break
    
    # Get movie indices and scores
    movie_indices = [i[0] for i in filtered_scores]
    scores = [i[1] for i in filtered_scores]
    
    # Build result DataFrame
    result = pd.DataFrame({
        'title': movies2.iloc[movie_indices]['title'].values,
        'similarity_score': scores
    })
    
    return result


def recommend_svd_for_user(user_id, top_n=10):
    """
    Recommend movies for a user using collaborative filtering (SVD).
    
    Predicts ratings for movies the user hasn't rated yet and returns
    top recommendations.
    
    Args:
        user_id (int): User ID to generate recommendations for.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame: Columns [movie_id, pred_rating, title, genre_text].
    """
    # Load SVD artifact
    artifact = load_svd_artifact()
    svd_model = artifact['svd']
    movies_lookup = artifact['movies_lookup']
    
    # Get movies user has already rated
    conn = sqlite3.connect(DB_PATH)
    rated_query = f"SELECT movie_id FROM ratings WHERE user_id = {user_id}"
    rated_movies = pd.read_sql_query(rated_query, conn)
    conn.close()
    
    rated_ids = set(rated_movies['movie_id'].values)
    
    # Get candidate movies (not yet rated)
    all_movie_ids = movies_lookup['movie_id'].values
    candidate_ids = np.setdiff1d(all_movie_ids, list(rated_ids))
    
    # Predict ratings for candidates
    predictions = []
    for movie_id in candidate_ids:
        try:
            pred = svd_model.predict(user_id, movie_id)
            predictions.append({
                'movie_id': movie_id,
                'pred_rating': pred.est
            })
        except Exception:
            # Skip movies that can't be predicted
            continue
    
    # Convert to DataFrame and sort
    if not predictions:
        return pd.DataFrame(columns=['movie_id', 'pred_rating', 'title', 'genre_text'])
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values('pred_rating', ascending=False).head(top_n)
    
    # Merge with movie metadata
    result = pred_df.merge(
        movies_lookup[['movie_id', 'title', 'genre_text']],
        on='movie_id',
        how='left'
    )
    
    return result


def recommend_content_topk(user_id, top_n=10, like_threshold=4.0):
    """
    Recommend movies using content-based filtering based on user's liked movies.
    
    Finds movies similar to those the user has rated highly (>= like_threshold).
    
    Args:
        user_id (int): User ID to generate recommendations for.
        top_n (int): Number of recommendations to return.
        like_threshold (float): Minimum rating to consider a movie "liked".
    
    Returns:
        pd.DataFrame: Columns [movie_id, content_score, title, genre_text].
    """
    content_data = build_content_recommender()
    movies2 = content_data['movies2']
    cosine_sim = content_data['cosine_sim']
    movieid_to_idx = content_data['movieid_to_idx']
    
    # Get user's liked movies
    conn = sqlite3.connect(DB_PATH)
    liked_query = f"""
    SELECT movie_id, rating 
    FROM ratings 
    WHERE user_id = {user_id} AND rating >= {like_threshold}
    """
    liked_movies = pd.read_sql_query(liked_query, conn)
    
    # Get all rated movies to exclude
    rated_query = f"SELECT movie_id FROM ratings WHERE user_id = {user_id}"
    rated_movies = pd.read_sql_query(rated_query, conn)
    conn.close()
    
    if liked_movies.empty:
        return pd.DataFrame(columns=['movie_id', 'content_score', 'title', 'genre_text'])
    
    rated_ids = set(rated_movies['movie_id'].values)
    liked_ids = liked_movies['movie_id'].values
    
    # Get indices of liked movies
    liked_indices = [movieid_to_idx[mid] for mid in liked_ids if mid in movieid_to_idx]
    
    if not liked_indices:
        return pd.DataFrame(columns=['movie_id', 'content_score', 'title', 'genre_text'])
    
    # Compute average similarity to liked movies
    avg_sim = cosine_sim[liked_indices].mean(axis=0)
    
    # Create candidate scores (exclude rated movies)
    candidate_scores = []
    for idx, score in enumerate(avg_sim):
        movie_id = movies2.iloc[idx]['movie_id']
        if movie_id not in rated_ids:
            candidate_scores.append({
                'movie_id': movie_id,
                'content_score': score,
                'title': movies2.iloc[idx]['title'],
                'genre_text': movies2.iloc[idx]['genre_text']
            })
    
    # Sort and return top N
    result = pd.DataFrame(candidate_scores)
    if result.empty:
        return result
    
    result = result.sort_values('content_score', ascending=False).head(top_n)
    return result


def safe_minmax(x):
    """
    Safely normalize array to [0, 1] range, avoiding division by zero.
    
    Args:
        x (np.ndarray): Array to normalize.
    
    Returns:
        np.ndarray: Normalized array.
    """
    x_range = x.ptp()  # peak-to-peak (max - min)
    if x_range == 0:
        return np.zeros_like(x)
    return (x - x.min()) / x_range


def hybrid_recommend_for_user(user_id, alpha=0.7, top_n=10, like_threshold=4.0):
    """
    Generate hybrid recommendations combining SVD and content-based filtering.
    
    Combines normalized SVD predictions and content similarity scores using:
    hybrid_score = alpha * svd_norm + (1 - alpha) * content_norm
    
    Args:
        user_id (int): User ID to generate recommendations for.
        alpha (float): Weight for SVD component (0 to 1).
        top_n (int): Number of recommendations to return.
        like_threshold (float): Minimum rating for content-based liked movies.
    
    Returns:
        pd.DataFrame: Columns [movie_id, hybrid_score, title, genre_text].
    """
    # Get SVD recommendations
    svd_recs = recommend_svd_for_user(user_id, top_n=100)
    
    # Get content recommendations
    content_recs = recommend_content_topk(user_id, top_n=100, like_threshold=like_threshold)
    
    if svd_recs.empty and content_recs.empty:
        return pd.DataFrame(columns=['movie_id', 'hybrid_score', 'title', 'genre_text'])
    
    # Merge on movie_id
    merged = svd_recs.merge(
        content_recs[['movie_id', 'content_score']],
        on='movie_id',
        how='outer'
    )
    
    # Fill missing values with 0
    merged['pred_rating'] = merged['pred_rating'].fillna(0)
    merged['content_score'] = merged['content_score'].fillna(0)
    
    # Normalize scores
    svd_norm = safe_minmax(merged['pred_rating'].values)
    content_norm = safe_minmax(merged['content_score'].values)
    
    # Compute hybrid score
    merged['hybrid_score'] = alpha * svd_norm + (1 - alpha) * content_norm
    
    # Sort and get top N
    merged = merged.sort_values('hybrid_score', ascending=False).head(top_n)
    
    # Get movie metadata if missing
    if 'title' not in merged.columns or merged['title'].isna().any():
        movies2 = load_movies_from_db()
        merged = merged.merge(
            movies2[['movie_id', 'title', 'genre_text']],
            on='movie_id',
            how='left',
            suffixes=('', '_new')
        )
        # Use new values if original was missing
        if 'title_new' in merged.columns:
            merged['title'] = merged['title'].fillna(merged['title_new'])
            merged['genre_text'] = merged['genre_text'].fillna(merged['genre_text_new'])
            merged = merged.drop(columns=['title_new', 'genre_text_new'])
    
    result = merged[['movie_id', 'hybrid_score', 'title', 'genre_text']]
    return result


if __name__ == "__main__":
    """
    Smoke test for recommendation functions.
    """
    print("=" * 60)
    print("CineAnalytica Recommender System - Smoke Test")
    print("=" * 60)
    
    # Check if required files exist
    missing_files = []
    if not DB_PATH.exists():
        missing_files.append(str(DB_PATH))
    if not (MODEL_DIR / "svd_recommender.joblib").exists():
        missing_files.append(str(MODEL_DIR / "svd_recommender.joblib"))
    if not (MODEL_DIR / "hybrid_params.joblib").exists():
        missing_files.append(str(MODEL_DIR / "hybrid_params.joblib"))
    
    if missing_files:
        print("\n⚠️  Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all model artifacts and database are present.")
    else:
        print("\n✓ All required files found\n")
        
        # Test content-based recommendation
        print("Test 1: Content-based recommendation for 'The Dark Knight'")
        print("-" * 60)
        try:
            recs = recommend_by_title("The Dark Knight", top_n=5)
            if isinstance(recs, dict) and 'error' in recs:
                print(f"Error: {recs['error']}")
                if recs.get('did_you_mean'):
                    print(f"Did you mean: {recs['did_you_mean'][:3]}")
            else:
                print(recs['title'].tolist())
        except Exception as e:
            print(f"Error: {e}")
        
        # Test hybrid recommendation
        print("\n\nTest 2: Hybrid recommendation for user_id=1")
        print("-" * 60)
        try:
            hybrid_recs = hybrid_recommend_for_user(user_id=1, top_n=5)
            if hybrid_recs.empty:
                print("No recommendations generated (user may not exist or have no ratings)")
            else:
                print(hybrid_recs.head())
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
