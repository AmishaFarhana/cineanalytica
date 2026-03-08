import streamlit as st
import sqlite3
import pandas as pd

DB_PATH = "data/cineanalytica.db"

# -----------------------------
# Helpers
# -----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(show_spinner=False)
def list_tables():
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
            conn,
        )
    return df["name"].tolist()

def has_table(name: str) -> bool:
    return name in list_tables()

def run_sql(query: str, params=None):
    if params is None:
        params = []
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)

@st.cache_data(show_spinner=False)
def get_homepage_stats():
    """Fetch homepage statistics from database"""
    query = """
    SELECT
        (SELECT COUNT(*) FROM movies) as total_movies,
        (SELECT COUNT(*) FROM ratings) as total_ratings,
        (SELECT COUNT(*) FROM reviews) as total_reviews,
        (SELECT COUNT(*) FROM actors) as total_actors,
        (SELECT COUNT(*) FROM directors) as total_directors,
        (SELECT ROUND(AVG(vote_average), 2) FROM movies WHERE vote_average > 0) as avg_rating
    """
    with get_conn() as conn:
        df = pd.read_sql_query(query, conn)
    return df.iloc[0].to_dict()

def build_filter_sql(selected_year, selected_genre):
    """
    Returns: (join_sql, where_sql, params)
    join_sql may include joins to genre tables only when needed.
    """
    joins = []
    wheres = []
    params = []

    # Year filter (assumes movies.release_date exists)
    if selected_year is not None:
        wheres.append("strftime('%Y', m.release_date) = ?")
        params.append(str(selected_year))

    # Genre filter (only if genre tables exist)
    if selected_genre is not None:
        # We assume common schema: movie_genres(movie_id, genre_id) + genres(genre_id, genre_name)
        if has_table("movie_genres") and has_table("genres"):
            joins.append("JOIN movie_genres mg ON mg.movie_id = m.movie_id")
            joins.append("JOIN genres g ON g.genre_id = mg.genre_id")
            wheres.append("g.genre_name = ?")
            params.append(selected_genre)
        else:
            # If schema differs, we can't apply genre filter safely
            st.warning("Genre filter selected, but tables `movie_genres` / `genres` not found. Skipping genre filter.")
            selected_genre = None

    join_sql = "\n".join(joins)
    where_sql = ""
    if wheres:
        where_sql = "WHERE " + " AND ".join(wheres)

    return join_sql, where_sql, params


# -----------------------------
# Streamlit UI - base
# -----------------------------
st.set_page_config(page_title="CineAnalytica", layout="wide")
st.title("🎬 CineAnalytica Dashboard")

# Homepage statistics
try:
    stats = get_homepage_stats()
    
    # Row 1: Total Movies, Avg Rating, Total Actors
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎬 Total Movies", f"{stats['total_movies']:,}")
    with col2:
        st.metric("⭐ Avg Rating", stats['avg_rating'])
    with col3:
        st.metric("🎭 Total Actors", f"{stats['total_actors']:,}")
    
    # Row 2: Total Directors, Total Ratings, Total Reviews
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("🎬 Total Directors", f"{stats['total_directors']:,}")
    with col5:
        st.metric("🗳️ Total Ratings", f"{stats['total_ratings']:,}")
    with col6:
        st.metric("💬 Total Reviews", f"{stats['total_reviews']:,}")
    
    st.divider()
except Exception as e:
    st.warning(f"Could not load homepage statistics: {e}")

st.markdown("### 🔍 Explore 5,000 movies · ML predictions · Sentiment analysis · Recommendations")

# Quick schema debug (super useful while building)
with st.expander("🧾 Debug: Show available tables (schema check)"):
    st.write(list_tables())

# -----------------------------
# Sidebar: Filters + Navigation
# -----------------------------
st.sidebar.header("Filters")

# Release years dropdown (optional)
year_options = [None]
try:
    if has_table("movies"):
        years_df = run_sql("""
            SELECT DISTINCT CAST(strftime('%Y', release_date) AS INT) AS release_year
            FROM movies
            WHERE release_date IS NOT NULL
            ORDER BY release_year DESC;
        """)
        year_options += [int(y) for y in years_df["release_year"].dropna().tolist()]
except Exception:
    pass

selected_year = st.sidebar.selectbox(
    "Release Year (optional)",
    options=year_options,
    format_func=lambda x: "All" if x is None else str(x),
)

# Genre dropdown (optional)
genre_options = [None]
try:
    if has_table("genres"):
        gdf = run_sql("SELECT DISTINCT genre_name FROM genres ORDER BY genre_name;")
        genre_options += gdf["genre_name"].dropna().tolist()
except Exception:
    pass

selected_genre = st.sidebar.selectbox(
    "Genre (optional)",
    options=genre_options,
    format_func=lambda x: "All" if x is None else str(x),
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Top Revenue", "Genre Trends", "Budget vs Revenue", "Top Actors/Directors", "SQL Query Viewer"],
)

# Pre-build filter SQL for pages that use movies
join_sql, where_sql, params = build_filter_sql(selected_year, selected_genre)

# -----------------------------
# Page: Top Revenue
# -----------------------------
if page == "Top Revenue":
    st.subheader("💰 Top 10 Movies by Revenue")

    if not has_table("movies"):
        st.error("Table `movies` not found in the database. Check your DB build step.")
        st.stop()

    query = f"""
    SELECT
        m.movie_id,
        m.title,
        m.revenue,
        m.vote_average
    FROM movies m
    {join_sql}
    {where_sql}
      {"AND" if where_sql else "WHERE"} m.revenue IS NOT NULL
    ORDER BY m.revenue DESC
    LIMIT 10;
    """

    df = run_sql(query, params=params)
    if df.empty:
        st.warning("No movies matched your filters. Try switching Genre/Year to 'All'.")
    st.dataframe(df, use_container_width=True)

    with st.expander("🔎 Show SQL used"):
        st.code(query)
        st.write("Params:", params)

# -----------------------------
# Page: Genre Trends
# -----------------------------
elif page == "Genre Trends":
    st.subheader("📈 Genre Trends Over Time")

    if not (has_table("movies") and has_table("movie_genres") and has_table("genres")):
        st.error("This page requires tables: `movies`, `movie_genres`, `genres`.")
        st.stop()

    query = """
    SELECT
        CAST(strftime('%Y', m.release_date) AS INT) AS release_year,
        g.genre_name,
        COUNT(DISTINCT m.movie_id) AS num_movies,
        AVG(m.revenue) AS avg_revenue
    FROM movies m
    JOIN movie_genres mg ON mg.movie_id = m.movie_id
    JOIN genres g ON g.genre_id = mg.genre_id
    WHERE m.release_date IS NOT NULL
    GROUP BY release_year, g.genre_name
    ORDER BY release_year, num_movies DESC;
    """
    df = run_sql(query)

    st.write("Pick a genre to visualize:")
    genre_pick = st.selectbox("Genre", sorted(df["genre_name"].unique().tolist()))
    plot_df = df[df["genre_name"] == genre_pick].sort_values("release_year")

    st.line_chart(plot_df.set_index("release_year")["num_movies"])
    st.dataframe(plot_df, use_container_width=True)

    with st.expander("🔎 Show SQL used"):
        st.code(query)

# -----------------------------
# Page: Budget vs Revenue
# -----------------------------
elif page == "Budget vs Revenue":
    st.subheader("🎯 Budget vs Revenue")

    if not has_table("movies"):
        st.error("Table `movies` not found.")
        st.stop()

    # budget column name might vary; try common names
    possible_budget_cols = ["budget", "production_budget"]
    movies_cols = set(run_sql("PRAGMA table_info(movies);")["name"].tolist())
    budget_col = next((c for c in possible_budget_cols if c in movies_cols), None)

    if budget_col is None:
        st.error("Could not find a budget column in `movies` (expected `budget` or `production_budget`).")
        st.stop()

    query = f"""
    SELECT
        m.movie_id,
        m.title,
        m.{budget_col} AS budget,
        m.revenue,
        m.vote_average
    FROM movies m
    WHERE m.{budget_col} IS NOT NULL
      AND m.revenue IS NOT NULL
      AND m.{budget_col} > 0
      AND m.revenue > 0
    ORDER BY m.revenue DESC
    LIMIT 2000;
    """
    df = run_sql(query)

    st.write("Scatter plot of budget vs revenue (top 2000 by revenue).")
    st.scatter_chart(df.set_index("movie_id")[["budget", "revenue"]])

    st.dataframe(df.head(50), use_container_width=True)

    with st.expander("🔎 Show SQL used"):
        st.code(query)

# -----------------------------
# Page: Top Actors/Directors
# -----------------------------
elif page == "Top Actors/Directors":
    st.subheader("🌟 Top Actors / Directors")

    col1, col2 = st.columns(2)

    # --- Actors ---
    with col1:
        st.markdown("#### 🎭 Top Actors by #Movies")

        # Try common actor-credit table names
        actor_table_candidates = ["movie_cast", "cast", "movie_actors"]
        actor_table = next((t for t in actor_table_candidates if has_table(t)), None)

        if actor_table is None or not has_table("actors"):
            st.info("Could not find actor credits table (`movie_cast`/`cast`/`movie_actors`) + `actors`.")
        else:
            # Try common column patterns
            # We'll attempt a query that works for the most common schema:
            # movie_cast(movie_id, actor_id, cast_order) + actors(actor_id, actor_name)
            query = f"""
            SELECT
                a.actor_name,
                COUNT(DISTINCT mc.movie_id) AS num_movies
            FROM {actor_table} mc
            JOIN actors a ON a.actor_id = mc.actor_id
            GROUP BY a.actor_name
            ORDER BY num_movies DESC
            LIMIT 10;
            """
            try:
                df = run_sql(query)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.warning(f"Actor query failed due to schema mismatch: {e}")

    # --- Directors ---
    with col2:
        st.markdown("#### 🎬 Top Directors by #Movies")

        director_table_candidates = ["movie_directors", "directing", "movie_crew_directors"]
        director_table = next((t for t in director_table_candidates if has_table(t)), None)

        if director_table is None or not has_table("directors"):
            st.info("Could not find director credits table (`movie_directors`/etc) + `directors`.")
        else:
            # Common schema: movie_directors(movie_id, director_id) + directors(director_id, director_name)
            query = f"""
            SELECT
                d.director_name,
                COUNT(DISTINCT md.movie_id) AS num_movies
            FROM {director_table} md
            JOIN directors d ON d.director_id = md.director_id
            GROUP BY d.director_name
            ORDER BY num_movies DESC
            LIMIT 10;
            """
            try:
                df = run_sql(query)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.warning(f"Director query failed due to schema mismatch: {e}")

# -----------------------------
# Page: SQL Query Viewer
# -----------------------------
elif page == "SQL Query Viewer":
    st.subheader("🧠 SQL Query Viewer")
    st.write("Write any SELECT query and run it safely.")

    default_sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    user_sql = st.text_area("Your SQL (SELECT only recommended):", value=default_sql, height=180)

    run = st.button("Run Query")

    if run:
        try:
            df = run_sql(user_sql)
            st.success(f"Returned {len(df)} rows × {len(df.columns)} columns")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Query failed: {e}")
