import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_ratings(path='data/ratings.csv', min_movie_ratings=5):
    ratings = pd.read_csv(path)
    
    # Drop timestamp
    ratings = ratings.drop(columns=['timestamp'])
    
    # Filter movies with enough ratings
    movie_counts = ratings['movieId'].value_counts()
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
    
    # Normalize ratings to 0-1
    scaler = MinMaxScaler()
    ratings['rating'] = scaler.fit_transform(ratings[['rating']])
    
    # Encode users and movies
    user_ids = ratings['userId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = ratings['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    
    ratings['user'] = ratings['userId'].map(user2user_encoded)
    ratings['movie'] = ratings['movieId'].map(movie2movie_encoded)
    
    movie_encoded2movie = {x: i for i, x in movie2movie_encoded.items()}
    
    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    
    # Convert rating to float32
    ratings['rating'] = ratings['rating'].values.astype(np.float32)
    
    return ratings, num_users, num_movies, user2user_encoded, movie2movie_encoded, movie_encoded2movie, scaler

def train_test_split_userwise(ratings, test_frac=0.2):
    train_rows, test_rows = [], []
    for user_id, user_data in ratings.groupby('user'):
        user_data = user_data.sample(frac=1, random_state=42)
        n_items = len(user_data)
        train_size = max(1, int((1 - test_frac) * n_items))
        test_size = max(1, n_items - train_size)
        train_rows.append(user_data.iloc[:train_size])
        test_rows.append(user_data.iloc[train_size:train_size+test_size])
    
    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)
    
    X_train = [train_df['user'].values, train_df['movie'].values]
    y_train = train_df['rating'].values
    
    X_test = [test_df['user'].values, test_df['movie'].values]
    y_test = test_df['rating'].values
    
    return X_train, y_train, X_test, y_test, train_df, test_df
