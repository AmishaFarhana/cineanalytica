"""
Database Tests

Validates that the SQLite database exists and contains required tables.
"""

import sqlite3
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "cineanalytica.db"


def test_database_exists():
    """Test that the database file exists."""
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}"


def test_database_connection():
    """Test that we can connect to the database."""
    conn = sqlite3.connect(DB_PATH)
    assert conn is not None
    conn.close()


def test_required_tables_exist():
    """Test that all required tables exist in the database."""
    required_tables = [
        'movies',
        'ratings',
        'movie_features',
        'genres',
        'movie_genres'
    ]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    
    conn.close()
    
    # Check each required table
    for table in required_tables:
        assert table in existing_tables, f"Required table '{table}' not found in database"


def test_movies_table_structure():
    """Test that movies table has expected columns."""
    expected_columns = {'movie_id', 'title'}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(movies)")
    columns = {row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    assert expected_columns.issubset(columns), \
        f"Movies table missing expected columns. Found: {columns}"


def test_ratings_table_structure():
    """Test that ratings table has expected columns."""
    expected_columns = {'user_id', 'movie_id', 'rating'}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(ratings)")
    columns = {row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    assert expected_columns.issubset(columns), \
        f"Ratings table missing expected columns. Found: {columns}"


def test_tables_not_empty():
    """Test that key tables contain data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check movies table
    cursor.execute("SELECT COUNT(*) FROM movies")
    movie_count = cursor.fetchone()[0]
    assert movie_count > 0, "Movies table is empty"
    
    # Check ratings table
    cursor.execute("SELECT COUNT(*) FROM ratings")
    rating_count = cursor.fetchone()[0]
    assert rating_count > 0, "Ratings table is empty"
    
    conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
