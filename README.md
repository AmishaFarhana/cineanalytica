# 🎬 CineAnalytica

> A full-stack movie analytics and recommendation platform built with Python, Streamlit, and Machine Learning.

[
[
[

***

## 🚀 Live Demo

👉 **[cineanalytica.streamlit.app](https://cineanalytica.streamlit.app)**

***

## 📌 Overview

CineAnalytica is a capstone project for MS Business Analytics at California State University East Bay. It combines exploratory data analysis, machine learning, and natural language processing to deliver a comprehensive movie intelligence platform — inspired by how Netflix thinks about content.

***

## ✨ Features

| Page | Description |
|------|-------------|
| 📊 **EDA Explorer** | Interactive visualizations of movie trends, genres, ratings, and revenue distributions |
| 🎯 **Box Office Predictor** | XGBoost model predicting movie revenue based on budget, genre, runtime, and more |
| 💬 **Sentiment Analyzer** | NLP-powered sentiment analysis of movie reviews using TF-IDF and SVD |
| 🎬 **Recommendation Engine** | Hybrid recommendation system combining content-based filtering and collaborative filtering (SVD) |

***

## 🧠 Models

| Model | File | Description |
|-------|------|-------------|
| XGBoost Regressor | `models/box_office_xgb.joblib` | Predicts box office revenue |
| Hybrid Recommender Params | `models/hybrid_params.joblib` | Alpha weights for hybrid blending |
| Sentiment Classifier | `models/sentiment_model.joblib` | TF-IDF + SVD sentiment pipeline |
| SVD Recommender | `models/svd_recommender.joblib` | Collaborative filtering via matrix factorization |

***

## 🗂️ Project Structure

```
cineanalytica/
├── app/
│   └── app.py                  # Main Streamlit application
├── data/                       # Datasets (CSV files)
├── models/                     # Trained .joblib model artifacts
├── notebooks/                  # Jupyter notebooks for EDA & model training
├── scripts/                    # Data processing & training scripts
├── tests/                      # Unit tests (pytest)
├── MODEL_CARD.md               # Model documentation
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version pin
└── README.md
```

***

## ⚙️ Installation & Local Setup

### Prerequisites
- Python 3.11
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AmishaFarhana/cineanalytica.git
cd cineanalytica

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/app.py
```

***

## 🛠️ Tech Stack

- **Frontend / UI**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost, scikit-surprise
- **NLP**: NLTK, TF-IDF
- **Data Processing**: pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
- **Version Control**: Git + GitHub
- **Deployment**: Streamlit Community Cloud

***

## 👥 Team

| Name | Role |
|------|------|
| Amisha Farhana Shaik | ML Engineering, Deployment, App Development |
| Khush Domadia | Data Analysis, Report, Model Development |

***

## 📚 Course

**MS Business Analytics Capstone Project**
California State University East Bay

***

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.