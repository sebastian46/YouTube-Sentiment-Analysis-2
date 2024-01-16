import re

def clean_text(text):
    """Lowercase and remove special characters"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def tokenize(text):
    """Tokenize the cleaned text"""
    return text.split()

# Example usage
def preprocess_comments(comments):
    preprocessed_comments = []
    for comment in comments:
        cleaned = clean_text(comment)
        tokens = tokenize(cleaned)
        preprocessed_comments.append(tokens)
    return preprocessed_comments
