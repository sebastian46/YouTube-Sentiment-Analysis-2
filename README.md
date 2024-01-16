# Sentiment Analysis on IMDB Dataset

This document outlines various model architectures experimented with for performing sentiment analysis on the IMDB movie reviews dataset.

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





