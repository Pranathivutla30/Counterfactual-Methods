import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Flatten, Concatenate, BatchNormalization
from tensorflow.keras.models import Model

def get_ncf_model(num_users, num_movies, embedding_size=32):
    # Inputs
    user_input = Input(shape=(1,), name="user_input")
    movie_input = Input(shape=(1,), name="movie_input")
    
    # Embeddings
    user_embedding = Embedding(num_users, embedding_size, embeddings_initializer="he_normal")(user_input)
    movie_embedding = Embedding(num_movies, embedding_size, embeddings_initializer="he_normal")(movie_input)
    
    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)
    
    # Concatenate embeddings
    concat = Concatenate()([user_vec, movie_vec])
    
    # Dense layers
    dense = Dense(256, activation='relu')(concat)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    
    dense = Dense(64, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    
    output = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
