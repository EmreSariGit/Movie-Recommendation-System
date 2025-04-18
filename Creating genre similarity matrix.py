import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the movies data
movies_df = pd.read_csv('C:/Users/Emre/Desktop/ml-latest-small/movies.csv')

# Prepare the genres column for vectorization
movies_df['genres'] = movies_df['genres'].fillna('')  # Handle any missing values

# Use CountVectorizer to vectorize the genres column
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))  # Split genres by '|'
genre_matrix = vectorizer.fit_transform(movies_df['genres'])  # Create genre matrix

# Calculate cosine similarity between movies
genre_similarity_matrix = cosine_similarity(genre_matrix, dense_output=True) 

# Save the dense matrix as a .npy file
np.save('genre_similarity_matrix.npy', genre_similarity_matrix)
print("Dense genre similarity matrix saved successfully!")