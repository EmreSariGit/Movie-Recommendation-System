from recommender import recommend

rated_movie_ids = [163134, 48780, 115680, 166291, 122920, 73881, 193583, 122916]
user_id = 611

recommend("collaborative", user_id)
recommend("genre", user_id)
recommend("mixed", user_id, alpha=0.5)
