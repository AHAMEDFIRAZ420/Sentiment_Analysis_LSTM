# -----------------------------------------------------------------------------
# Sentiment Analysis using a Single LSTM Network
# This script is a refactored version designed for higher accuracy and stability.
# It uses the IMDb movie review dataset and a robust text preprocessing pipeline.
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import re

# --- 1. Load and Preprocess the Dataset ---

print("Loading IMDb dataset...")
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

print(f"Number of training samples: {len(x_train)}")
print(f"Number of testing samples: {len(x_test)}")
print("Example of a raw review (integer-encoded):", x_train[0][:10])

# LSTM models require input sequences to have the same length.
max_review_length = 500
print(f"\nPadding sequences to a maximum length of {max_review_length}...")
x_train = pad_sequences(x_train, maxlen=max_review_length)
x_test = pad_sequences(x_test, maxlen=max_review_length)

print("Shape of training data after padding:", x_train.shape)
print("Shape of testing data after padding:", x_test.shape)

# --- 2. Build the Single LSTM Model ---

print("\nBuilding the single LSTM model...")
embedding_vector_length = 64
lstm_units = 128

model = Sequential()

model.add(Embedding(input_dim=num_words, output_dim=embedding_vector_length, input_length=max_review_length))

# Dropout Layer: Regularization to prevent overfitting.
model.add(Dropout(0.2))

# Single LSTM Layer: The core of the model for learning sequence patterns.
model.add(LSTM(units=lstm_units))

# Final Dense Layer: A fully connected layer for binary classification.
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a standard learning rate
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

# --- 3. Train the Model ---
print("\nTraining the model with Early Stopping...")
epochs = 25
batch_size = 64


early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping_callback],
                    verbose=2)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --- 4. Evaluate the Model on the Test Set ---
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

# --- 5. Make a Prediction on a New Review ---
def preprocess_text(review_text):
    """Preprocesses a new review to be compatible with the model."""
    # Build the word index from the training data for consistency
    word_index = imdb.get_word_index()
    word_to_int = {key: (value + 3) for key, value in word_index.items()}
    word_to_int["<PAD>"] = 0
    word_to_int["<START>"] = 1
    word_to_int["<UNK>"] = 2
    
    review_text = review_text.lower()
    review_text = re.sub(r'[^\w\s]', '', review_text)
    words = review_text.split()
    
    # Convert words to integers, using the unknown token for unseen words
    review_integers = [word_to_int.get(word, 2) for word in words]
    
    # Pad the sequence
    padded_review = pad_sequences([review_integers], maxlen=max_review_length)
    
    return padded_review

def predict_sentiment(review_text):
    """Predicts the sentiment of a new text review."""
    processed_review = preprocess_text(review_text)
    prediction = model.predict(processed_review)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    print(f"\nReview: '{review_text}'")
    print(f"Prediction score: {prediction:.4f}")
    print(f"Predicted sentiment: {sentiment}")
    
    return sentiment, prediction

# Start an interactive session for user input
print("\n--- Interactive Prediction ---")
print("Enter a movie review below. Type 'quit' to exit.")

while True:
    user_input = input("Enter a review: ")
    if user_input.lower() == 'quit':
        print("Exiting interactive prediction.")
        break
    
    predict_sentiment(user_input)
