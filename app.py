import os
import pickle
from googleapiclient.discovery import build
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from data_preprocessor import preprocess_comments, clean_text


# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

# Ensure API key is loaded
if not API_KEY:
    print("API key not found. Please check your .env file.")
    exit()

# Load the trained model and word index
model = load_model('./models')
with open('./models/word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)

# Set up YouTube service object
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Function to fetch comments from a video
def get_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()

    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

# Function to get video details
def get_video_details(video_id):
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()

    return response

# Function to encode text using the word index
def encode_text(text, word_index, max_length=250):
    # Preprocess and tokenize the text
    tokens = clean_text(text).split()
    # Replace words with indices
    tokens = [1] + [word_index.get(word, 2) for word in tokens]
    # Pad sequences
    return pad_sequences([tokens], maxlen=max_length, padding='post')

# Function to predict sentiment
def predict_sentiment(text, model, word_index):
    encoded_text = encode_text(text, word_index)
    prediction = model.predict(encoded_text)
    return "Positive" if prediction[0][0] > 0.5 else "Negative"

# Replace 'VIDEO_ID' with a YouTube video ID
video_id = 'zh008MNMOlo'
# video_details = get_video_details(video_id)
comments = get_comments(video_id)

# Predict sentiment for each comment
for comment in comments:
    sentiment = predict_sentiment(comment, model, word_index)
    print(f"Review: {comment}\nSentiment: {sentiment}\n")
