# Sentiment Analysis on IMDB Dataset

This document outlines various model architectures experimented with for performing sentiment analysis on the IMDB movie reviews dataset.

## How to use
### Step 1: Obtain a YouTube Data API v3 Key
1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Navigate to "APIs & Services" > "Dashboard" and click "+ ENABLE APIS AND SERVICES".
4. Search for "YouTube Data API v3" and enable it.
5. Go to "Credentials" in the left sidebar, click "Create credentials", and choose "API key". Your new API key will be displayed.
6. Store this API key in a `.env` file in your project's root directory:
```bash
YOUTUBE_API_KEY=your_api_key_here
```

### Step 2: Set Up the Python Environment
1. Ensure Python is installed on your system. This was developed on 3.10 for tensorflow.
2. Install the required Python packages:
```bash
pip install tensorflow google-api-python-client python-dotenv
```

### Step 3: Train the Sentiment Analysis Model
1. Run model.py to train the model on the IMDB dataset:
```bash
python model.py
```
2. The model and word index are saved in the /models directory.

### Step 4: Train the Sentiment Analysis Model
1. Change the video_id variable in app.py to the ID of your desired YouTube video:
```bash
video_id = 'desired_youtube_video_id_here'
```
2. Run app.py to fetch and analyze comments:
```bash
python app.py
```
3. The script outputs the sentiment (Positive/Negative) for each comment.


## Model Architectures

### 1. Basic LSTM Model

#### Architecture
- Embedding Layer
- LSTM Layer
- Dense Output Layer (Sigmoid Activation)

#### Performance
- High training accuracy, poor generalization to test data.
- Overfitting observed.

#### Analysis
- Effective at capturing long-range dependencies.
- Prone to overfitting due to lack of regularization.

---

### 2. Bidirectional LSTM with Regularization

#### Architecture
- Embedding Layer
- Bidirectional LSTM Layer
- Dropout
- Global Average Pooling
- Dense Layers (ReLU and Sigmoid Activations)
- L2 Regularization

#### Performance
- High training accuracy, moderate test accuracy.
- Predictions biased towards one class.

#### Analysis
- Bidirectional LSTM captures patterns effectively.
- Overfitting reduced but still present.

---

### 3. Simplified Model with GlobalAveragePooling1D

#### Architecture
- Embedding Layer
- GlobalAveragePooling1D
- Dense Layers (ReLU and Sigmoid Activations)

#### Performance
- Balanced accuracy on training and test sets.
- Correctly classified both positive and negative reviews.

#### Analysis
- Simpler architecture led to better generalization.
- GlobalAveragePooling1D reduced complexity and overfitting.
- Demonstrated robustness and capacity to learn relevant patterns.

---

## Conclusion

Through various experiments, it was found that a simpler model architecture could outperform more complex models on the IMDB dataset. The key takeaway is that complex models, while powerful, can easily overfit and may not always provide the best solution for a given problem, especially in cases where the data does not require highly complex representations. The simpler model with GlobalAveragePooling1D offered a better balance between learning capacity and generalization, leading to improved performance on unseen data.





