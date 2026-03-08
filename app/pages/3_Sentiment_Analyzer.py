import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="wide")
st.title("😊 Movie Review Sentiment Analyzer")

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cineanalytica.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'sentiment_model.joblib')

@st.cache_resource
def load_sentiment_model():
    """Load the trained sentiment model"""
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def search_movies(query):
    """Search movies by title"""
    try:
        conn = sqlite3.connect(DB_PATH)
        sql = """
        SELECT movie_id, title, overview, vote_average, popularity
        FROM movies
        WHERE title LIKE ? AND overview IS NOT NULL AND overview != ''
        ORDER BY popularity DESC
        LIMIT 20
        """
        df = pd.read_sql_query(sql, conn, params=[f"%{query}%"])
        conn.close()
        return df, sql
    except Exception as e:
        st.error(f"Error searching movies: {e}")
        return pd.DataFrame(), ""

@st.cache_data
def get_all_movies_with_overview():
    """Fetch all movies with overview text"""
    try:
        conn = sqlite3.connect(DB_PATH)
        sql = """
        SELECT movie_id, title, overview, vote_average
        FROM movies
        WHERE overview IS NOT NULL AND overview != ''
        """
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, sql
    except Exception as e:
        st.error(f"Error fetching movies: {e}")
        return pd.DataFrame(), ""

@st.cache_data
def get_genre_sentiment_data():
    """Fetch movies with genres for sentiment analysis"""
    try:
        conn = sqlite3.connect(DB_PATH)
        sql = """
        SELECT 
            m.movie_id,
            m.title,
            m.overview,
            g.genre_name
        FROM movies m
        JOIN movie_genres mg ON m.movie_id = mg.movie_id
        JOIN genres g ON mg.genre_id = g.genre_id
        WHERE m.overview IS NOT NULL AND m.overview != ''
        """
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, sql
    except Exception as e:
        st.error(f"Error fetching genre data: {e}")
        return pd.DataFrame(), ""

# Load model
model, error = load_sentiment_model()

if error:
    st.error(f"⚠️ Failed to load sentiment model: {error}")
    st.info("Please ensure the model file exists at: " + MODEL_PATH)
    st.stop()

st.success("✅ Sentiment model loaded successfully!")

# Section 1: Movie Search and Single Movie Sentiment
st.header("🔍 Single Movie Sentiment Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Search Movie")
    
    search_query = st.text_input(
        "Enter movie title",
        placeholder="e.g., The Dark Knight",
        help="Search for a movie by title"
    )
    
    selected_movie = None
    movie_overview = None
    
    if search_query:
        search_results, search_sql = search_movies(search_query)
        
        if not search_results.empty:
            movie_options = {f"{row['title']} ({row['vote_average']:.1f}⭐)": row 
                           for _, row in search_results.iterrows()}
            
            selected_title = st.selectbox(
                "Select a movie",
                options=list(movie_options.keys()),
                help="Choose from search results"
            )
            
            if selected_title:
                selected_movie = movie_options[selected_title]
                movie_overview = selected_movie['overview']
                
                st.write("**Movie Overview:**")
                st.info(movie_overview)
                
                with st.expander("View SQL used"):
                    st.code(search_sql, language="sql")
        else:
            st.warning("No movies found. Try a different search term.")

with col2:
    st.subheader("Sentiment Analysis Result")
    
    if movie_overview:
        try:
            # Predict sentiment
            prediction = model.predict([movie_overview])[0]
            probabilities = model.predict_proba([movie_overview])[0]
            
            # Determine sentiment label and confidence
            # Assuming binary classification: 0=Negative, 1=Positive
            sentiment_label = "Positive" if prediction == 1 else "Negative"
            confidence = probabilities[prediction]
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = probabilities[1] - probabilities[0]
            
            # Display results
            st.metric("Sentiment", sentiment_label, delta=f"{confidence*100:.1f}% confidence")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sentiment_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentiment Score"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "lightcoral"},
                        {'range': [-0.3, 0.3], 'color': "lightyellow"},
                        {'range': [0.3, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show probability breakdown
            st.write("**Probability Breakdown:**")
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Positive'],
                'Probability': [probabilities[0], probabilities[1]]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error analyzing sentiment: {e}")
            st.exception(e)
    else:
        st.info("👈 Search and select a movie to analyze its sentiment")

# Section 2: Batch Sentiment Analysis
st.header("📊 Batch Sentiment Analysis")

try:
    all_movies, all_movies_sql = get_all_movies_with_overview()
    
    if not all_movies.empty:
        with st.spinner("Analyzing sentiment for all movies..."):
            # Predict sentiment for all movies
            predictions = model.predict(all_movies['overview'].tolist())
            probabilities = model.predict_proba(all_movies['overview'].tolist())
            
            # Add predictions to dataframe
            all_movies['sentiment'] = predictions
            all_movies['sentiment_label'] = all_movies['sentiment'].apply(
                lambda x: 'Positive' if x == 1 else 'Negative'
            )
            all_movies['confidence'] = [probs[pred] for probs, pred in zip(probabilities, predictions)]
            all_movies['sentiment_score'] = probabilities[:, 1] - probabilities[:, 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of sentiment distribution
            sentiment_counts = all_movies['sentiment_label'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution',
                color=sentiment_counts.index,
                color_discrete_map={'Positive': 'lightgreen', 'Negative': 'lightcoral'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            st.metric("Total Movies Analyzed", len(all_movies))
            st.metric("Positive Sentiment", f"{(all_movies['sentiment'] == 1).sum()} ({(all_movies['sentiment'] == 1).sum() / len(all_movies) * 100:.1f}%)")
            st.metric("Negative Sentiment", f"{(all_movies['sentiment'] == 0).sum()} ({(all_movies['sentiment'] == 0).sum() / len(all_movies) * 100:.1f}%)")
        
        with col2:
            # Top 5 most confidently positive
            st.subheader("🌟 Top 5 Most Positive Movies")
            top_positive = all_movies[all_movies['sentiment'] == 1].nlargest(5, 'confidence')[
                ['title', 'confidence', 'vote_average']
            ].copy()
            top_positive['confidence'] = top_positive['confidence'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(top_positive, use_container_width=True, hide_index=True)
            
            # Top 5 most confidently negative
            st.subheader("😞 Top 5 Most Negative Movies")
            top_negative = all_movies[all_movies['sentiment'] == 0].nlargest(5, 'confidence')[
                ['title', 'confidence', 'vote_average']
            ].copy()
            top_negative['confidence'] = top_negative['confidence'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(top_negative, use_container_width=True, hide_index=True)
        
        with st.expander("View SQL used"):
            st.code(all_movies_sql, language="sql")
            
    else:
        st.warning("No movies with overview text found in database")
        
except Exception as e:
    st.error(f"Error in batch analysis: {e}")
    st.exception(e)

# Section 3: Genre Sentiment Breakdown
st.header("🎭 Sentiment by Genre")

try:
    genre_data, genre_sql = get_genre_sentiment_data()
    
    if not genre_data.empty:
        with st.spinner("Analyzing sentiment by genre..."):
            # Predict sentiment for genre data
            genre_predictions = model.predict(genre_data['overview'].tolist())
            genre_probabilities = model.predict_proba(genre_data['overview'].tolist())
            
            # Add predictions
            genre_data['sentiment_score'] = genre_probabilities[:, 1] - genre_probabilities[:, 0]
            
            # Calculate average sentiment by genre
            genre_sentiment = genre_data.groupby('genre_name')['sentiment_score'].agg([
                ('avg_sentiment', 'mean'),
                ('count', 'count')
            ]).reset_index()
            
            genre_sentiment = genre_sentiment.sort_values('avg_sentiment', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            genre_sentiment,
            x='avg_sentiment',
            y='genre_name',
            orientation='h',
            title='Average Sentiment Score by Genre',
            labels={'avg_sentiment': 'Average Sentiment Score', 'genre_name': 'Genre'},
            color='avg_sentiment',
            color_continuous_scale=['red', 'yellow', 'green'],
            hover_data=['count']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("View Genre Sentiment Data"):
            display_df = genre_sentiment.copy()
            display_df['avg_sentiment'] = display_df['avg_sentiment'].apply(lambda x: f"{x:.3f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with st.expander("View SQL used"):
            st.code(genre_sql, language="sql")
            
    else:
        st.warning("No genre data available")
        
except Exception as e:
    st.error(f"Error in genre analysis: {e}")
    st.exception(e)

# Additional info
st.divider()
st.info("💡 **Note:** The sentiment model analyzes movie overview text to predict whether the description has positive or negative sentiment. This is based on the language used in the description, not actual movie reviews.")
