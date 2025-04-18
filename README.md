# 🎬 Movie Recommendation System

A hybrid movie recommender system using Collaborative Filtering (CF) and Content-Based Filtering (CBF), based on the MovieLens dataset. The system predicts user preferences and suggests relevant movies by combining user behavior and genre similarity.

## 📌 Features

- **Collaborative Filtering (CF)** using Surprise SVD model
- **Content-Based Filtering (CBF)** using genre similarity between movies
- **Hybrid Recommendation** by combining CF and CBF scores
- Optional filtering using **movie popularity**
- Efficient filtering to avoid already-rated movies

🧠 How It Works
CF (Collaborative Filtering) uses the Surprise library's trained SVD model to predict user ratings for unrated movies.

CBF (Content-Based Filtering) calculates genre similarity from a precomputed matrix.

Mixed Recommendation blends CF and CBF using a configurable alpha (default: 0.6).

Results can also be sorted by popularity.

📊 Dataset
Based on MovieLens Latest Small Dataset

Genre similarity matrix precomputed with cosine similarity
