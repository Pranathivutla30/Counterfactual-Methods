import matplotlib.pyplot as plt
from src.preprocessing import load_ratings, train_test_split_userwise
from src.model import get_ncf_model
import tensorflow as tf

# Load data
ratings, num_users, num_movies, user2user_encoded, movie2movie_encoded, movie_encoded2movie, scaler = load_ratings('data/ratings.csv')

# Train-test split
X_train, y_train, X_test, y_test, train_df, test_df = train_test_split_userwise(ratings)

# Initialize model
model = get_ncf_model(num_users, num_movies, embedding_size=32)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("ncf_best_model.h5", save_best_only=True)
]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# Save loss plot
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig('results/loss_plot.png')
plt.show()