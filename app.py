import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

# Ensure API key is loaded
if not API_KEY:
    print("API key not found. Please check your .env file.")
    exit()

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

# Replace 'VIDEO_ID' with a YouTube video ID
video_id = 'dQw4w9WgXcQ'
video_details = get_video_details(video_id)
comments = get_comments(video_id)
# print(len(comments))
for comment in comments:
    print(comment)

# print(video_details)
