import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Page config
st.set_page_config(page_title="Recommendation Engine", page_icon="🎯", layout="wide")
st.title("🎯 Movie Recommendation Engine")

# Paths
DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'cineanalytica.db'
MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'svd_recommender.joblib'

@st.cache_resource
def load_recommender_model():
    """Load the SVD recommender model bundle"""
    try:
        bundle = joblib.load(MODEL_PATH)
        svd = bundle['svd']
        movies_lookup = bundle['movies_lookup']
        return svd, movies_lookup, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def build_content_recommender(_movies_df):
    """Build TF-IDF content-based recommender"""
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=50000, ngram_range=(1, 2))
        mat = tfidf.fit_transform(_movies_df['genre_text'].fillna(''))
        cosine_sim = linear_kernel(mat, mat)
        title_to_idx = {t: i for i, t in enumerate(_movies_df['title'])}
        return cosine_sim, title_to_idx, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def get_all_genres():
    """Fetch all unique genres"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        sql = "SELECT DISTINCT genre_name FROM genres ORDER BY genre_name"
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df['genre_name'].tolist(), sql
    except Exception as e:
        st.error(f"Error fetching genres: {e}")
        return [], ""

@st.cache_data
def get_top_active_users(limit=20):
    """Get most active users by rating count"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        sql = f"""
        SELECT user_id, COUNT(*) as rating_count
        FROM ratings
        GROUP BY user_id
        ORDER BY rating_count DESC
        LIMIT {limit}
        """
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, sql
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return pd.DataFrame(), ""

@st.cache_data
def get_user_ratings(user_id):
    """Get all ratings for a specific user"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        sql = """
        SELECT r.movie_id, r.rating, m.title
        FROM ratings r
        JOIN movies m ON r.movie_id = m.movie_id
        WHERE r.user_id = ?
        """
        df = pd.read_sql_query(sql, conn, params=[user_id])
        conn.close()
        return df, sql
    except Exception as e:
        st.error(f"Error fetching user ratings: {e}")
        return pd.DataFrame(), ""

@st.cache_data
def get_movies_by_genre(genre_name):
    """Get top movies by genre"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        sql = """
        SELECT DISTINCT
            m.movie_id,
            m.title,
            m.vote_average,
            m.popularity,
            m.revenue
        FROM movies m
        JOIN movie_genres mg ON m.movie_id = mg.movie_id
        JOIN genres g ON mg.genre_id = g.genre_id
        WHERE g.genre_name = ?
        AND m.vote_average > 0
        ORDER BY m.vote_average DESC
        LIMIT 10
        """
        df = pd.read_sql_query(sql, conn, params=[genre_name])
        conn.close()
        return df, sql
    except Exception as e:
        st.error(f"Error fetching movies by genre: {e}")
        return pd.DataFrame(), ""

@st.cache_data
def get_hidden_gems():
    """Get hidden gem movies"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        sql = """
        SELECT 
            m.title,
            m.vote_average,
            m.popularity,
            m.revenue,
            m.release_date,
            GROUP_CONCAT(DISTINCT g.genre_name) as genres
        FROM movies m
        LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.genre_id
        WHERE m.vote_average > 7.5 
        AND m.popularity < 10
        GROUP BY m.movie_id, m.title, m.vote_average, m.popularity, m.revenue, m.release_date
        ORDER BY m.vote_average DESC
        LIMIT 20
        """
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, sql
    except Exception as e:
        st.error(f"Error fetching hidden gems: {e}")
        return pd.DataFrame(), ""

# Load models
svd, movies_lookup, model_error = load_recommender_model()

if model_error:
    st.error(f"⚠️ Failed to load recommender model: {model_error}")
    st.info("Please ensure the model file exists at: " + str(MODEL_PATH))
    st.stop()

st.success("✅ Recommendation models loaded successfully!")

# Build content-based recommender
cosine_sim, title_to_idx, content_error = build_content_recommender(movies_lookup)

if content_error:
    st.warning(f"⚠️ Content-based recommender not available: {content_error}")

# Section 1: Content-Based Recommender
st.header("🎬 Content-Based Recommendations")
st.write("Find similar movies based on genre and content features")

if cosine_sim is not None and title_to_idx is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Movie selection
        movie_titles = sorted(movies_lookup['title'].tolist())
        selected_movie = st.selectbox(
            "Select a movie",
            options=movie_titles,
            help="Choose a movie to get similar recommendations"
        )
        
        if selected_movie:
            st.write(f"**Selected:** {selected_movie}")
            
            # Get genre info if available
            movie_info = movies_lookup[movies_lookup['title'] == selected_movie]
            if not movie_info.empty and 'genre_text' in movie_info.columns:
                st.write(f"**Genres:** {movie_info.iloc[0]['genre_text']}")
    
    with col2:
        if selected_movie:
            try:
                # Get movie index
                idx = title_to_idx[selected_movie]
                
                # Get similarity scores
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                # Get top 11 (excluding the movie itself)
                sim_scores = sim_scores[1:11]
                
                # Get movie indices and scores
                movie_indices = [i[0] for i in sim_scores]
                similarity_scores = [i[1] for i in sim_scores]
                
                # Create recommendations dataframe
                recommendations = movies_lookup.iloc[movie_indices].copy()
                recommendations['similarity_score'] = similarity_scores
                
                # Display as bar chart
                fig = px.bar(
                    recommendations,
                    x='similarity_score',
                    y='title',
                    orientation='h',
                    title=f'Top 10 Similar Movies to "{selected_movie}"',
                    labels={'similarity_score': 'Similarity Score', 'title': 'Movie Title'},
                    color='similarity_score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show table
                with st.expander("View Recommendations Table"):
                    display_df = recommendations[['title', 'similarity_score']].copy()
                    display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                st.exception(e)
else:
    st.warning("Content-based recommender not available")

# Section 2: Collaborative Filtering (SVD)
st.header("👥 Collaborative Filtering Recommendations")
st.write("Personalized recommendations based on user rating patterns")

if svd is not None:
    # Get top active users
    active_users, users_sql = get_top_active_users(20)
    
    if not active_users.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # User selection
            user_options = {f"User {row['user_id']} ({row['rating_count']} ratings)": row['user_id'] 
                          for _, row in active_users.iterrows()}
            
            selected_user_label = st.selectbox(
                "Select a user",
                options=list(user_options.keys()),
                help="Choose a user to get personalized recommendations"
            )
            
            selected_user_id = user_options[selected_user_label]
            
            st.write(f"**Selected User ID:** {selected_user_id}")
            
            # Show user's existing ratings
            user_ratings, user_ratings_sql = get_user_ratings(selected_user_id)
            
            if not user_ratings.empty:
                st.write(f"**Movies Rated:** {len(user_ratings)}")
                st.write(f"**Average Rating:** {user_ratings['rating'].mean():.2f}")
                
                with st.expander("View User's Ratings"):
                    st.dataframe(user_ratings[['title', 'rating']].head(10), use_container_width=True, hide_index=True)
        
        with col2:
            if st.button("🎯 Generate Recommendations", type="primary"):
                try:
                    with st.spinner("Generating personalized recommendations..."):
                        # Get movies user hasn't rated
                        rated_movie_ids = set(user_ratings['movie_id'].tolist())
                        all_movie_ids = set(movies_lookup['movie_id'].tolist())
                        unrated_movie_ids = list(all_movie_ids - rated_movie_ids)
                        
                        # Predict ratings for unrated movies
                        predictions = []
                        for movie_id in unrated_movie_ids:
                            pred = svd.predict(selected_user_id, movie_id)
                            predictions.append({
                                'movie_id': movie_id,
                                'predicted_rating': pred.est
                            })
                        
                        # Convert to dataframe and sort
                        pred_df = pd.DataFrame(predictions)
                        pred_df = pred_df.sort_values('predicted_rating', ascending=False).head(10)
                        
                        # Merge with movie titles
                        pred_df = pred_df.merge(movies_lookup[['movie_id', 'title']], on='movie_id')
                        
                        # Display results
                        st.success(f"✅ Top 10 Recommendations for User {selected_user_id}")
                        
                        # Create bar chart
                        fig = px.bar(
                            pred_df,
                            x='predicted_rating',
                            y='title',
                            orientation='h',
                            title=f'Predicted Ratings for User {selected_user_id}',
                            labels={'predicted_rating': 'Predicted Rating', 'title': 'Movie Title'},
                            color='predicted_rating',
                            color_continuous_scale='RdYlGn',
                            range_color=[0, 5]
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show table
                        display_df = pred_df[['title', 'predicted_rating']].copy()
                        display_df['predicted_rating'] = display_df['predicted_rating'].apply(lambda x: f"{x:.2f}")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    st.exception(e)
        
        with st.expander("View SQL used"):
            st.code(users_sql, language="sql")
    else:
        st.warning("No user data available")
else:
    st.warning("SVD model not available")

# Section 3: Popular by Genre
st.header("🏆 Top Movies by Genre")
st.write("Discover the highest-rated movies in each genre")

all_genres, genres_sql = get_all_genres()

if all_genres:
    selected_genre = st.selectbox(
        "Select a genre",
        options=all_genres,
        help="Choose a genre to see top-rated movies"
    )
    
    if selected_genre:
        genre_movies, genre_movies_sql = get_movies_by_genre(selected_genre)
        
        if not genre_movies.empty:
            # Create horizontal bar chart
            fig = px.bar(
                genre_movies,
                x='vote_average',
                y='title',
                orientation='h',
                title=f'Top 10 {selected_genre} Movies by Rating',
                labels={'vote_average': 'Average Rating', 'title': 'Movie Title'},
                color='vote_average',
                color_continuous_scale='Viridis',
                hover_data=['popularity', 'revenue']
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            with st.expander("View Data Table"):
                display_df = genre_movies[['title', 'vote_average', 'popularity']].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with st.expander("View SQL used"):
                st.code(genre_movies_sql, language="sql")
        else:
            st.warning(f"No movies found for genre: {selected_genre}")
else:
    st.warning("No genres available")

# Section 4: Hidden Gems
st.header("💎 Hidden Gems")
st.write("High-quality movies with low popularity - underrated treasures!")

hidden_gems, hidden_gems_sql = get_hidden_gems()

if not hidden_gems.empty:
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hidden Gems Found", len(hidden_gems))
    with col2:
        st.metric("Average Rating", f"{hidden_gems['vote_average'].mean():.2f}")
    with col3:
        st.metric("Average Popularity", f"{hidden_gems['popularity'].mean():.2f}")
    
    # Create scatter plot
    fig = px.scatter(
        hidden_gems,
        x='popularity',
        y='vote_average',
        hover_data=['title', 'genres', 'release_date'],
        title='Hidden Gems: High Rating, Low Popularity',
        labels={'popularity': 'Popularity Score', 'vote_average': 'Average Rating'},
        color='vote_average',
        color_continuous_scale='Viridis',
        size='vote_average',
        size_max=15
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display table
    st.subheader("Complete List of Hidden Gems")
    display_df = hidden_gems[['title', 'vote_average', 'popularity', 'genres', 'release_date']].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with st.expander("View SQL used"):
        st.code(hidden_gems_sql, language="sql")
else:
    st.warning("No hidden gems found")

# Additional info
st.divider()
st.info("💡 **Tip:** Content-based recommendations use movie features like genres, while collaborative filtering learns from user rating patterns. Try both approaches to discover new movies!")
