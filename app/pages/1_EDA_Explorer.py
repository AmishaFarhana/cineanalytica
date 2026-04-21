import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="EDA Explorer", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

st.markdown("*Explore trends, budgets, revenues and star power across 5,000 movies using interactive SQL-backed charts.*")
st.divider()

with st.expander("ℹ️ How to use this page"):
    st.markdown("""
- Use the **sidebar filters** to narrow by year range and genre
- Scroll down to explore Genre Trends, Budget vs Revenue scatter, Star Power rankings, and Monthly Heatmap
- Click **View SQL used** under any chart to see the exact query behind it
""")

# Database connection
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cineanalytica.db')

@st.cache_data
def get_connection():
    """Create database connection"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data
def get_all_genres():
    """Fetch all unique genres"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT DISTINCT genre_name FROM genres ORDER BY genre_name"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['genre_name'].tolist()

@st.cache_data
def get_year_range():
    """Get min and max years from movies"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        MIN(CAST(strftime('%Y', release_date) AS INTEGER)) as min_year,
        MAX(CAST(strftime('%Y', release_date) AS INTEGER)) as max_year
    FROM movies
    WHERE release_date IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return int(df['min_year'].iloc[0]), int(df['max_year'].iloc[0])

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Year range slider
min_year, max_year = get_year_range()
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Genre multiselect
all_genres = get_all_genres()
selected_genres = st.sidebar.multiselect(
    "Select Genres",
    options=all_genres,
    default=all_genres
)

# Build filter conditions
genre_filter = f"AND g.genre_name IN ({','.join(['?' for _ in selected_genres])})" if selected_genres else ""
year_filter = "AND CAST(strftime('%Y', m.release_date) AS INTEGER) BETWEEN ? AND ?"

# Section 1: Top 10 Movies by Revenue
st.header("💰 Top 10 Movies by Revenue")

@st.cache_data
def get_top_movies(genres, year_start, year_end):
    """Fetch top 10 movies by revenue"""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT DISTINCT
        m.title,
        m.budget,
        m.revenue,
        m.release_date,
        m.vote_average,
        GROUP_CONCAT(DISTINCT g.genre_name) as genres
    FROM movies m
    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
    LEFT JOIN genres g ON mg.genre_id = g.genre_id
    WHERE m.revenue > 0
    {year_filter}
    {'AND g.genre_name IN (' + ','.join(['?' for _ in genres]) + ')' if genres else ''}
    GROUP BY m.movie_id, m.title, m.budget, m.revenue, m.release_date, m.vote_average
    ORDER BY m.revenue DESC
    LIMIT 10
    """
    params = [year_start, year_end] + (genres if genres else [])
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    sql_display = query.replace('?', '{}').format(year_start, year_end, *[f"'{g}'" for g in genres] if genres else [])
    return df, sql_display

top_movies_df, top_movies_sql = get_top_movies(selected_genres, year_range[0], year_range[1])

if not top_movies_df.empty:
    # Format currency columns
    top_movies_df['budget'] = top_movies_df['budget'].apply(lambda x: f"${x:,.0f}")
    top_movies_df['revenue'] = top_movies_df['revenue'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top_movies_df, use_container_width=True)
else:
    st.warning("No data available for selected filters")

with st.expander("View SQL used"):
    st.code(top_movies_sql, language="sql")

# Section 2: Genre Trends Over Time
st.header("📈 Genre Trends Over Time")

@st.cache_data
def get_genre_trends(genres, year_start, year_end):
    """Fetch movie count per genre per year"""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT 
        CAST(strftime('%Y', m.release_date) AS INTEGER) as year,
        g.genre_name,
        COUNT(DISTINCT m.movie_id) as movie_count
    FROM movies m
    JOIN movie_genres mg ON m.movie_id = mg.movie_id
    JOIN genres g ON mg.genre_id = g.genre_id
    WHERE m.release_date IS NOT NULL
    {year_filter}
    {'AND g.genre_name IN (' + ','.join(['?' for _ in genres]) + ')' if genres else ''}
    GROUP BY year, g.genre_name
    ORDER BY year, g.genre_name
    """
    params = [year_start, year_end] + (genres if genres else [])
    
    with st.spinner("Loading genre trends..."):
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    
    sql_display = query.replace('?', '{}').format(year_start, year_end, *[f"'{g}'" for g in genres] if genres else [])
    return df, sql_display

try:
    genre_trends_df, genre_trends_sql = get_genre_trends(selected_genres, year_range[0], year_range[1])

    if not genre_trends_df.empty:
        fig = px.line(
            genre_trends_df,
            x='year',
            y='movie_count',
            color='genre_name',
            title='Number of Movies Released by Genre Over Time',
            labels={'year': 'Year', 'movie_count': 'Number of Movies', 'genre_name': 'Genre'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for selected filters")

    with st.expander("View SQL used"):
        st.code(genre_trends_sql, language="sql")
        
except Exception as e:
    st.error(f"Error loading genre trends: {e}")

# Section 3: Budget vs Revenue Scatter
st.header("💵 Budget vs Revenue Analysis")

@st.cache_data
def get_budget_revenue(genres, year_start, year_end):
    """Fetch budget and revenue data with genre"""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT DISTINCT
        m.title,
        m.budget,
        m.revenue,
        g.genre_name,
        m.vote_average
    FROM movies m
    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
    LEFT JOIN genres g ON mg.genre_id = g.genre_id
    WHERE m.budget > 0 AND m.revenue > 0
    {year_filter}
    {'AND g.genre_name IN (' + ','.join(['?' for _ in genres]) + ')' if genres else ''}
    """
    params = [year_start, year_end] + (genres if genres else [])
    
    with st.spinner("Loading budget vs revenue data..."):
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    
    sql_display = query.replace('?', '{}').format(year_start, year_end, *[f"'{g}'" for g in genres] if genres else [])
    return df, sql_display

try:
    budget_revenue_df, budget_revenue_sql = get_budget_revenue(selected_genres, year_range[0], year_range[1])

    if not budget_revenue_df.empty:
        fig = px.scatter(
            budget_revenue_df,
            x='budget',
            y='revenue',
            color='genre_name',
            hover_data=['title', 'vote_average'],
            title='Budget vs Revenue by Genre',
            labels={'budget': 'Budget ($)', 'revenue': 'Revenue ($)', 'genre_name': 'Genre'},
            opacity=0.6,
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for selected filters")

    with st.expander("View SQL used"):
        st.code(budget_revenue_sql, language="sql")
        
except Exception as e:
    st.error(f"Error loading budget vs revenue data: {e}")

# Section 4: Top Actors by Star Power (Window Function Showcase)
st.header("🎭 Top Actors by Star Power")

@st.cache_data
def get_top_actors_with_rank(genres, year_start, year_end):
    """Fetch top actors by average revenue using window function"""
    conn = sqlite3.connect(DB_PATH)
    
    # Build the base query with window function
    base_query = """
    SELECT 
        a.actor_name,
        COUNT(mc.movie_id) as num_movies,
        AVG(m.revenue) as avg_revenue,
        RANK() OVER (ORDER BY AVG(m.revenue) DESC) as revenue_rank
    FROM actors a
    JOIN movie_cast mc ON a.actor_id = mc.actor_id
    JOIN movies m ON mc.movie_id = m.movie_id
    WHERE m.revenue > 0
    """
    
    # Add filters if needed
    if genres or (year_start and year_end):
        # Need to add joins for filtering
        query = """
        SELECT 
            a.actor_name,
            COUNT(DISTINCT m.movie_id) as num_movies,
            AVG(m.revenue) as avg_revenue,
            RANK() OVER (ORDER BY AVG(m.revenue) DESC) as revenue_rank
        FROM actors a
        JOIN movie_cast mc ON a.actor_id = mc.actor_id
        JOIN movies m ON mc.movie_id = m.movie_id
        LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.genre_id
        WHERE m.revenue > 0
        """
        
        # Add year filter
        if year_start and year_end:
            query += f" AND CAST(strftime('%Y', m.release_date) AS INTEGER) BETWEEN ? AND ?"
        
        # Add genre filter
        if genres:
            query += f" AND g.genre_name IN ({','.join(['?' for _ in genres])})"
        
        query += """
        GROUP BY a.actor_id, a.actor_name
        HAVING num_movies >= 3
        ORDER BY avg_revenue DESC
        LIMIT 20
        """
        
        params = []
        if year_start and year_end:
            params.extend([year_start, year_end])
        if genres:
            params.extend(genres)
    else:
        query = base_query + """
        GROUP BY a.actor_id, a.actor_name
        HAVING COUNT(mc.movie_id) >= 3
        ORDER BY avg_revenue DESC
        LIMIT 20
        """
        params = []
    
    with st.spinner("Loading top actors with window function..."):
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    
    # Format SQL for display
    sql_display = query
    if params:
        for param in params:
            if isinstance(param, str):
                sql_display = sql_display.replace('?', f"'{param}'", 1)
            else:
                sql_display = sql_display.replace('?', str(param), 1)
    
    return df, sql_display

try:
    top_actors_df, top_actors_sql = get_top_actors_with_rank(selected_genres, year_range[0], year_range[1])

    if not top_actors_df.empty:
        # Format currency columns for display
        display_df = top_actors_df.copy()
        display_df['avg_revenue'] = display_df['avg_revenue'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available for selected filters")

    with st.expander("Window Function SQL"):
        st.code(top_actors_sql, language="sql")
        
except Exception as e:
    st.error(f"Error loading actor data: {e}")

# Section 5: Monthly Release Heatmap
st.header("📅 Monthly Release Patterns")

@st.cache_data
def get_monthly_revenue(genres, year_start, year_end):
    """Fetch average revenue by month"""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT 
        CAST(strftime('%m', m.release_date) AS INTEGER) as month,
        COUNT(*) as num_releases,
        AVG(m.revenue) as avg_revenue
    FROM movies m
    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
    LEFT JOIN genres g ON mg.genre_id = g.genre_id
    WHERE m.release_date IS NOT NULL AND m.revenue > 0
    {year_filter}
    {'AND g.genre_name IN (' + ','.join(['?' for _ in genres]) + ')' if genres else ''}
    GROUP BY month
    ORDER BY month
    """
    params = [year_start, year_end] + (genres if genres else [])
    
    with st.spinner("Loading monthly release patterns..."):
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    
    # Add month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['month_name'] = df['month'].apply(lambda x: month_names[x-1] if 1 <= x <= 12 else '')
    
    sql_display = query.replace('?', '{}').format(year_start, year_end, *[f"'{g}'" for g in genres] if genres else [])
    return df, sql_display

try:
    monthly_revenue_df, monthly_revenue_sql = get_monthly_revenue(selected_genres, year_range[0], year_range[1])

    if not monthly_revenue_df.empty:
        fig = px.bar(
            monthly_revenue_df,
            x='month_name',
            y='avg_revenue',
            title='Average Revenue by Release Month',
            labels={'month_name': 'Month', 'avg_revenue': 'Average Revenue ($)'},
            color='num_releases',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show movie count per month
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Most Profitable Month", 
                      monthly_revenue_df.loc[monthly_revenue_df['avg_revenue'].idxmax(), 'month_name'])
        with col2:
            st.metric("Most Releases", 
                      monthly_revenue_df.loc[monthly_revenue_df['num_releases'].idxmax(), 'month_name'])
    else:
        st.warning("No data available for selected filters")

    with st.expander("View SQL used"):
        st.code(monthly_revenue_sql, language="sql")
        
except Exception as e:
    st.error(f"Error loading monthly release data: {e}")
