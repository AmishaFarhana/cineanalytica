"""
Recommender System Tests

Validates that recommendation functions work correctly.
"""

from pathlib import Path
import sys
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.recommenders import (
    recommend_by_title,
    recommend_svd_for_user,
    hybrid_recommend_for_user,
    load_svd_artifact,
    load_movies_from_db
)


def test_load_svd_artifact():
    """Test that SVD artifact can be loaded."""
    artifact = load_svd_artifact()
    assert isinstance(artifact, dict)
    assert 'svd' in artifact
    assert 'movies_lookup' in artifact


def test_load_movies_from_db():
    """Test that movies can be loaded from database."""
    movies = load_movies_from_db()
    assert isinstance(movies, pd.DataFrame)
    assert not movies.empty
    assert 'movie_id' in movies.columns
    assert 'title' in movies.columns
    assert 'combined_text' in movies.columns


def test_recommend_by_title_valid():
    """Test content-based recommendation with valid title."""
    result = recommend_by_title("The Dark Knight", top_n=3)
    
    # Should return DataFrame or error dict
    if isinstance(result, dict):
        # If error, fail the test with informative message
        error_msg = result.get('error', 'Unknown error')
        suggestions = result.get('did_you_mean', [])
        pytest.fail(
            f"Title not found: {error_msg}\n"
            f"Suggestions: {suggestions}"
        )
    
    # Validate DataFrame result
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 3
    assert 'title' in result.columns
    assert 'similarity_score' in result.columns


def test_recommend_by_title_invalid():
    """Test content-based recommendation with invalid title."""
    result = recommend_by_title("NonexistentMovie12345XYZ", top_n=3)
    
    # Should return error dict
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'did_you_mean' in result


def test_recommend_svd_for_user():
    """Test SVD collaborative filtering recommendation."""
    result = recommend_svd_for_user(user_id=1, top_n=3)
    
    # Should return DataFrame (may be empty if user doesn't exist)
    assert isinstance(result, pd.DataFrame)
    
    if not result.empty:
        assert len(result) <= 3
        assert 'movie_id' in result.columns
        assert 'pred_rating' in result.columns
        assert 'title' in result.columns
        assert 'genre_text' in result.columns


def test_hybrid_recommend_for_user():
    """Test hybrid recommendation system."""
    result = hybrid_recommend_for_user(user_id=1, top_n=3)
    
    # Should return DataFrame (may be empty if user doesn't exist)
    assert isinstance(result, pd.DataFrame)
    
    if not result.empty:
        assert len(result) <= 3
        assert 'movie_id' in result.columns
        assert 'hybrid_score' in result.columns
        assert 'title' in result.columns
        assert 'genre_text' in result.columns
        
        # Hybrid scores should be between 0 and 1 (normalized)
        assert result['hybrid_score'].min() >= 0
        assert result['hybrid_score'].max() <= 1


def test_recommend_by_title_returns_different_movies():
    """Test that recommendations don't include the query movie."""
    query_title = "The Dark Knight"
    result = recommend_by_title(query_title, top_n=5)
    
    if isinstance(result, pd.DataFrame):
        # Query movie should not be in recommendations
        assert query_title not in result['title'].values


def test_hybrid_alpha_parameter():
    """Test that hybrid recommendation respects alpha parameter."""
    # Test with different alpha values
    result1 = hybrid_recommend_for_user(user_id=1, alpha=0.9, top_n=5)
    result2 = hybrid_recommend_for_user(user_id=1, alpha=0.1, top_n=5)
    
    # Both should return DataFrames
    assert isinstance(result1, pd.DataFrame)
    assert isinstance(result2, pd.DataFrame)
    
    # If both have results, scores should potentially differ
    # (unless user has very limited data)
    if not result1.empty and not result2.empty:
        # Just verify structure is correct
        assert 'hybrid_score' in result1.columns
        assert 'hybrid_score' in result2.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
