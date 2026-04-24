import numpy as np
from src.preprocessing import load_ratings
import tensorflow as tf

def recommend_movies(user_id, model, user2user_encoded, movie2movie_encoded, movie_encoded2movie, train_df, num_movies, top_n=10):
    user_enc = user2user_encoded[user_id]
    all_movie_ids = np.arange(num_movies)
    movies_rated = train_df[train_df['user'] == user_enc]['movie'].values
    movies_to_predict = np.setdiff1d(all_movie_ids, movies_rated)
    user_array = np.full(len(movies_to_predict), user_enc)
    preds = model.predict([user_array, movies_to_predict], verbose=0).flatten()
    top_indices = movies_to_predict[np.argsort(preds)[::-1][:top_n]]
    recommended_movie_ids = [movie_encoded2movie[i] for i in top_indices]
    return recommended_movie_ids

if __name__ == "__main__":
    ratings, num_users, num_movies, user2user_encoded, movie2movie_encoded, movie_encoded2movie, scaler = load_ratings('data/ratings.csv')
    
    import pandas as pd
    # Train-test split
    from src.preprocessing import train_test_split_userwise
    X_train, y_train, X_test, y_test, train_df, test_df = train_test_split_userwise(ratings)
    
    # Load trained model
    model = tf.keras.models.load_model("ncf_best_model.h5")
    
    user_id = 1
    top_movies = recommend_movies(
        user_id, model, user2user_encoded, movie2movie_encoded, movie_encoded2movie, train_df, num_movies, top_n=10
    )
    print(f"Top 10 recommended movies for user {user_id}: {top_movies}")
