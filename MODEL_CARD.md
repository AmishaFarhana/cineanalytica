# CineAnalytica Model Card

## Model Overview

CineAnalytica is a comprehensive movie analytics system that combines multiple machine learning models to provide:

1. Box Office Revenue Prediction (XGBoost Regression)
2. Sentiment Analysis for Movie Reviews (Text Classification)
3. Movie Recommendations (Hybrid: SVD + Content-Based)

## Purpose

This system helps users:
- Predict potential box office revenue for upcoming movies
- Analyze sentiment of movie reviews
- Discover personalized movie recommendations based on viewing history and preferences

## Data Sources

### Primary Datasets
- **TMDB 5000 Movies Dataset**: Movie metadata including budget, revenue, cast, crew, genres
- **TMDB 5000 Credits Dataset**: Cast and crew information
- **MovieLens Dataset**: User ratings and tags
- **IMDB Dataset**: Movie reviews for sentiment analysis

### Data Statistics
- Movies: ~5,000 titles with complete metadata
- Ratings: User-movie interaction data
- Reviews: Text reviews with sentiment labels

## Preprocessing

### Box Office Model
- Log transformation of revenue and budget to handle skewed distributions
- Feature engineering: release year, release month, star power metrics
- Handling missing values: median imputation for numerical features
- Outlier detection and treatment for extreme budget/revenue values

### Sentiment Model
- Text cleaning: lowercase, punctuation removal, stopword filtering
- Tokenization and vectorization (TF-IDF or Count Vectorizer)
- Label encoding: positive/negative sentiment

### Recommendation System
- User-item rating matrix construction
- Content features: TF-IDF on combined text (title + overview + genres)
- SVD matrix factorization for collaborative filtering
- Cosine similarity computation for content-based filtering

## Features

### Box Office Prediction Features
- `budget`: Production budget (log-transformed)
- `runtime`: Movie duration in minutes
- `popularity`: TMDB popularity score
- `vote_average`: Average user rating (0-10)
- `vote_count`: Number of user votes
- `release_year`: Year of release
- `release_month`: Month of release (1-12)
- `log_star_power`: (Optional) Log-transformed star power metric based on cast

### Sentiment Analysis Features
- TF-IDF vectors from review text
- N-grams (unigrams and bigrams)
- Stop words removed

### Recommendation Features
- **Collaborative Filtering**: User-movie rating matrix (SVD decomposition)
- **Content-Based**: TF-IDF vectors from movie metadata (title, overview, genres)
- **Hybrid**: Weighted combination of both approaches

## Model Architecture

### 1. Box Office Revenue Predictor
- **Algorithm**: XGBoost Regressor
- **Target**: Log-transformed revenue (log1p)
- **Hyperparameters**: Tuned via cross-validation
- **Output**: Predicted log revenue (converted back to dollars using expm1)

### 2. Sentiment Analyzer
- **Algorithm**: Logistic Regression or Naive Bayes with TF-IDF
- **Target**: Binary sentiment (positive/negative)
- **Vectorization**: TF-IDF with max_features tuning
- **Output**: Sentiment label and confidence probability

### 3. Recommendation Engine
- **SVD (Collaborative Filtering)**: 
  - Matrix factorization using Surprise library
  - Predicts user ratings for unseen movies
  
- **Content-Based Filtering**:
  - TF-IDF vectorization (max_features=50,000, ngram_range=(1,2))
  - Cosine similarity between movies
  
- **Hybrid System**:
  - Combines normalized SVD predictions and content similarity
  - Formula: `hybrid_score = alpha * svd_norm + (1 - alpha) * content_norm`
  - Default alpha: 0.7 (70% collaborative, 30% content)

## Performance Metrics

### Box Office Model
- **Metric**: Root Mean Squared Error (RMSE) on log-transformed revenue
- **Evaluation**: 80/20 train-test split with cross-validation
- **Baseline**: Mean prediction benchmark

### Sentiment Model
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Evaluation**: Stratified train-test split
- **Baseline**: Majority class classifier

### Recommendation System
- **Metrics**: RMSE, MAE for rating prediction
- **Evaluation**: Leave-one-out cross-validation
- **Ranking Metrics**: Precision@K, Recall@K, NDCG

## Limitations

### General Limitations
- Models trained on historical data (pre-2017) may not capture recent trends
- Limited to English-language movies and reviews
- Performance degrades for movies with sparse metadata or ratings

### Box Office Model
- Cannot predict impact of marketing campaigns or social media buzz
- Doesn't account for competition from other releases
- Limited to theatrical revenue (excludes streaming, home video)
- Star power metric may be incomplete or outdated

### Sentiment Model
- May struggle with sarcasm, irony, and nuanced opinions
- Trained on IMDB reviews, may not generalize to other platforms
- Binary classification doesn't capture sentiment intensity

### Recommendation System
- Cold start problem: Poor recommendations for new users/movies
- Popularity bias: May over-recommend popular movies
- Filter bubble: Limited diversity in recommendations
- Requires sufficient rating history for accurate predictions

## Ethical Considerations

### Bias and Fairness
- Dataset may reflect historical biases in movie production and distribution
- Recommendation system may perpetuate existing preferences and limit exposure to diverse content
- Sentiment analysis trained on IMDB may reflect demographic biases of reviewers

### Privacy
- User rating data is anonymized
- No personally identifiable information (PII) is stored or processed
- Recommendations are generated locally without external data sharing

### Transparency
- Model predictions should be presented with confidence intervals or uncertainty estimates
- Users should be informed that recommendations are algorithmic and may not reflect personal taste
- System should allow users to provide feedback and adjust recommendations

### Intended Use
- Educational and research purposes
- Movie industry analysis and trend exploration
- Personal movie discovery and recommendation
- NOT intended for financial investment decisions
- NOT intended as sole basis for production decisions

## Reproduction

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
1. Place raw datasets in `data/raw/`:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`
   - `movies.csv`, `ratings.csv`, `tags.csv` (MovieLens)
   - `IMDB Dataset.csv`

2. Build SQLite database:
```bash
jupyter notebook notebooks/01_build_sqlite_db.ipynb
```

### Model Training

#### Box Office Model
```bash
jupyter notebook notebooks/02_box_office_model.ipynb
```
Output: `models/box_office_xgb.joblib`

#### Sentiment Model
```bash
jupyter notebook notebooks/03_sentiment_model.ipynb
```
Output: `models/sentiment_model.joblib`

#### Recommendation System
```bash
jupyter notebook notebooks/04_recommendation_engine.ipynb
```
Output: `models/svd_recommender.joblib`, `models/hybrid_params.joblib`

### Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test suites:
```bash
pytest tests/test_db.py -v
pytest tests/test_models.py -v
pytest tests/test_recommenders.py -v
```

### Inference

#### Box Office Prediction
```bash
python scripts/predict_box_office.py \
  --budget 100000000 \
  --runtime 120 \
  --popularity 50 \
  --vote_average 7.5 \
  --vote_count 12000 \
  --release_year 2012 \
  --release_month 7 \
  --log_star_power 17
```

#### Sentiment Analysis
```bash
python scripts/predict_sentiment.py --text "This movie was amazing!"
```

#### Recommendations
```bash
# Content-based
python scripts/recommend.py --title "The Dark Knight" --top_n 5

# Collaborative filtering
python scripts/recommend.py --user_id 1 --top_n 10

# Hybrid
python scripts/recommend.py --user_id 1 --hybrid --top_n 10
```

### Streamlit Application
```bash
streamlit run app/app.py
```

## Model Artifacts

All trained models are saved in the `models/` directory:
- `box_office_xgb.joblib`: XGBoost box office predictor
- `sentiment_model.joblib`: Sentiment classification pipeline
- `svd_recommender.joblib`: SVD collaborative filtering model + movie lookup
- `hybrid_params.joblib`: Hybrid recommendation parameters (alpha, etc.)

## Version History

- **v1.0** (Initial Release): Basic models with core functionality
- Future improvements:
  - Deep learning models for sentiment analysis
  - Neural collaborative filtering for recommendations
  - Real-time data updates and model retraining
  - Multi-language support

## Contact & Maintenance

For questions, issues, or contributions, please refer to the project repository.

**Last Updated**: March 2026
