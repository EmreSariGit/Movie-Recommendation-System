import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the data
ratings_df = pd.read_csv('data/ratings.csv')

# Drop the timestamp column since it’s not needed
ratings_df = ratings_df.drop('timestamp', axis=1)

# Im adding myself as a new user
new_ratings = pd.DataFrame({
    'userId': [611] * 8,  # New user ID
    'movieId': [163134, 48780, 115680, 166291, 122920, 73881, 193583, 122916],  # Movie IDs
    'rating': [5, 5, 3, 3.5, 3.5, 2.5, 4.5, 4]  # Corresponding ratings
})

# Append new ratings to the existing dataset
ratings_df = pd.concat([ratings_df, new_ratings], ignore_index=True)

# Load the data into the Surprise format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()

# Use SVD for collaborative filtering
model = SVD()
model.fit(trainset)

# Save the trained model
import pickle
with open('CollaborativeModel-ml-latest-small.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Function to recommend top-N movies for a user
def recommend_movies(user_id, model, ratings_df, n=10):
    # Get a list of all movieIds
    movie_ids = ratings_df['movieId'].unique()
    
    # Filter out movies the user has already rated
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unrated_movies = [movie_id for movie_id in movie_ids if movie_id not in rated_movies]
    
    # Predict ratings for each unrated movie
    recommendations = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in unrated_movies]
    
    # Sort the list by predicted ratings in descending order and select the top N
    top_n_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
    
    return top_n_recommendations

# Example usage: recommend top 10 movies for user with userId = 611
user_id = 611
top_movies = recommend_movies(user_id, model, ratings_df, n=10)
print("Recommended Movies:", top_movies)
