import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Define max_length used during training
max_length = 250  # Make sure this is the same length used in training

# Load the saved word index
with open('./models/word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)

# Function to encode new text using the word index
def encode_text(text, word_index, max_length):
    # Convert the text to lowercase and split into words
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    
    # Replace words not in the word_index with the index for "<UNK>"
    tokens = [1] + [word_index.get(word, 2) for word in tokens]
    
    # Pad sequences to the same length
    return pad_sequences([tokens], maxlen=max_length, padding='post')

# Function to predict sentiment of new text
def predict_sentiment(text, word_index, model, max_length):
    # Encode the text using the word index
    encoded_text = encode_text(text, word_index, max_length)
    # print("Encoded text:", encoded_text)
    
    # Predict sentiment
    prediction = model.predict(encoded_text)
    print(prediction)

    # Interpret the prediction
    return "Positive" if prediction[0][0] > 0.5 else "Negative"

# Load the trained model
model = load_model('./models')

# Test new sentences
test_sentences = [
    "This movie was fantastic! I really enjoyed it.",
    "This movie was not fantastic! I really did not enjoy it.",
    "I love this great movie!",
    "What a terrible movie, I wasted two hours of my life.",
    "The cinematography was breathtaking and added depth to the story.",
    "The plot was predictable and lacked originality.",
    "The lead actor's performance was incredibly powerful and moving.",
    "I found the movie to be dull and uninspiring.",
    "This is a masterpiece of filmmaking, with a brilliant script and outstanding acting.",
    "The movie's pacing was off and it felt dragged out.",
    "Bad, bad, bad",
    "negative, bad, terrible",
]

# test_sentences = [
#     "The sunset by the beach was breathtakingly beautiful.",
#     "I had a terrible day at work today.",
#     "The meal was delicious and the service was outstanding.",
#     "I'm so disappointed with the product I received.",
#     "This is the best book I've read in years!",
#     "The weather today is gloomy and depressing.",
#     "I'm incredibly grateful for all your help and support.",
#     "That's the worst customer service I've ever experienced.",
#     "The concert last night was an amazing experience.",
#     "I feel sick after eating at that restaurant.",
#     "My vacation was a truly relaxing and joyful experience.",
#     "This new phone model is incredibly fast and has great features.",
#     "I'm upset because my package arrived late and damaged.",
#     "The city park is so serene and well-maintained.",
#     "I've never dealt with such an unprofessional company.",
#     "The freshly baked bread from the bakery is heavenly.",
#     "My internet connection keeps dropping, it's so frustrating.",
#     "The medical staff at the clinic were very kind and attentive.",
#     "My coffee this morning was cold and tasted stale.",
#     "The new art exhibit at the museum is quite impressive."
# ]


for text in test_sentences:
    print(f"Review: {text}\nSentiment: {predict_sentiment(text, word_index, model, max_length)}\n")
