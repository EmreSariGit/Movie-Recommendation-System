import numpy as np
import pandas as pd
import pickle

popularity = pd.read_csv('data/popularity.csv', index_col='movieId').to_dict()['popularity']

# Load the trained model
with open('CollaborativeModel-ml-latest-small.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the movies data
movies_df = pd.read_csv('data/movies.csv')

# Load the genre similarity matrix
genre_similarity_matrix = np.load('genre_similarity_matrix.npy')

# Function to recommend top-N movies based on genres (CBF)
def recommend_by_genre(type, user_rated_movie_ids, genre_similarity_matrix, movies_df, n=10):
    # Find indices of user-rated movies
    rated_indices = movies_df[movies_df['movieId'].isin(user_rated_movie_ids)].index.tolist()
    
    # Compute average similarity scores for unrated movies
    genre_scores = genre_similarity_matrix[rated_indices].mean(axis=0)
    unrated_indices = [i for i in range(len(movies_df)) if movies_df.iloc[i]['movieId'] not in user_rated_movie_ids]
    recommendations = [(movies_df.iloc[i]['movieId'], genre_scores[i]) for i in unrated_indices]
    if type == "mixed":
        return recommendations
    if type == "genre":
        # Sort by similarity score and return top-N
        top_n_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
        return top_n_recommendations

# Function to recommend top-N movies based on user similarity (CF)
def recommend_movies(type, user_id, model, movies_df, rated_movie_ids, n=10):
    # Filter out movies the user has already rated
    unrated_movies = [movie_id for movie_id in movies_df['movieId'] if movie_id not in rated_movie_ids]
    
    # Predict ratings for each unrated movie
    recommendations = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in unrated_movies]
    if type == "mixed":
        return recommendations
    
    if type == "collaborative": 
        # Sort the list by predicted ratings in descending order and select the top N
        top_n_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
        return top_n_recommendations

# Function to blend the CF and CBF results
def mix_recommendations(user_id, alpha=0.6, n=10):
    cf_recommendations = recommend_movies("mixed", user_id, loaded_model, movies_df, rated_movie_ids)
    cbf_recommendations = recommend_by_genre("mixed", rated_movie_ids, genre_similarity_matrix, movies_df)
    # Normalize CF scores to [0, 1]
    cf_scores = {movie_id: (score - 1) / 4 for movie_id, score in cf_recommendations}
    
    # Create a combined recommendation list
    final_scores = []
    
    # Merge CF and CBF recommendations, adjusting the CF scores
    for movie_id, cbf_score in cbf_recommendations:
        cf_score = cf_scores.get(movie_id, 0)  # Default to 0 if CF score doesn't exist
        final_score = alpha * cf_score + (1 - alpha) * cbf_score
        final_scores.append((movie_id, final_score))
    
    # Sort recommendations by final score and return top-N
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    
    return final_scores[:n]  # Return top-N recommendations based on final score

# The function
def recommend(type, user_id, alpha=0.6, usepopularity="no"):
    n = 10
    if usepopularity == "yes":
        n = 1000
        
    if type == "mixed":
        top_movies = mix_recommendations(user_id, alpha, n)
        print("\nRecommended Top 10 Movies by Both Genre Similarity and User Similarity:")
    if type == "genre":
        top_movies = recommend_by_genre(type, rated_movie_ids, genre_similarity_matrix, movies_df, n)
        print("\nRecommended Top 10 Movies by Genre Similarity:")
    if type == "collaborative":
        top_movies = recommend_movies(type, user_id, loaded_model, movies_df, rated_movie_ids, n)
        print("\nRecommended Top 10 Movies by User Similarity:")

    if usepopularity == "yes":
        top_movies = sorted(top_movies, key=lambda x: popularity.get(x[0], 0), reverse=True)[:10]
            
    movie_ids = [movie_id for movie_id, _ in top_movies]
    movie_details = movies_df[movies_df['movieId'].isin(movie_ids)]
    recommended_movies_details = movie_details[['movieId', 'title', 'genres']]
    print(recommended_movies_details[['movieId', 'title', 'genres']].to_string(index=False))
    return recommended_movies_details

# Example usage 
rated_movie_ids = [163134, 48780, 115680, 166291, 122920, 73881, 193583, 122916] 
user_id = 611  

recommend("collaborative", 611)
recommend("genre", 611)
recommend("mixed", 611, 0.5)
