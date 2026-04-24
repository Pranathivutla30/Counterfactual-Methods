# src/evaluate.py
import numpy as np
from sklearn.metrics import mean_squared_error
from src.preprocessing import load_ratings, train_test_split_userwise
import tensorflow as tf
import os

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Load data
ratings, num_users, num_movies, user2user_encoded, movie2movie_encoded, movie_encoded2movie, scaler = load_ratings('data/ratings.csv')

# Train-test split
X_train, y_train, X_test, y_test, train_df, test_df = train_test_split_userwise(ratings)

# Load trained model
model = tf.keras.models.load_model("ncf_best_model.h5")

# Evaluate
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
y_pred = model.predict(X_test, verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Prepare metrics text
metrics_text = (
    f"Test Loss (MSE): {test_loss:.4f}\n"
    f"Test MAE: {test_mae:.4f}\n"
    f"Test RMSE: {rmse:.4f}\n"
)

# Save metrics to file
with open("results/metrics.txt", "w") as f:
    f.write(metrics_text)
