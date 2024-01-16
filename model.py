import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D, Bidirectional, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2

# Parameters for the model
VOCAB_SIZE = 10000  # Same as num_words in the dataset loading step
EMBEDDING_DIM = 16  # Embedding dimension
LSTM_UNITS = 32     # LSTM units
MAX_LEN = 250       # Maximum length of sequences
FILTERS = 32        # Number of filters for the Conv1D layer
KERNEL_SIZE = 3     # Kernel size for the Conv1D layer

# Load the IMDB dataset
# num_words specifies the number of words to keep based on their frequency
# The dataset is split into training and testing sets
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Get the word index from the IMDB dataset
word_index = tf.keras.datasets.imdb.get_word_index()
# Adjust the word index
adjusted_word_index = {word: (index + 3) for word, index in word_index.items() if index < VOCAB_SIZE - 3}
adjusted_word_index["<PAD>"] = 0
adjusted_word_index["<START>"] = 1
adjusted_word_index["<UNK>"] = 2

# Save the word index
with open('./models/word_index.pickle', 'wb') as handle:
    pickle.dump(adjusted_word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Pad the sequences to have uniform length
train_data = pad_sequences(train_data, maxlen=MAX_LEN)
test_data = pad_sequences(test_data, maxlen=MAX_LEN)

def create_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=5, validation_split=0.2)
    model.save('./models')

    return model

# Create the model
model = create_model()

results = model.evaluate(test_data, test_labels)
print(results)
