# TweetNLP Sentiment Analysis (From Scratch)

This project implements sentiment analysis on tweets using deep learning models built from scratch using PyTorch.

---

## 📌 Objective

- Build a baseline model for tweet sentiment classification
- Improve the model architecture without using pretrained models
- Compare performance using standard evaluation metrics

---

## 🧠 Models Implemented

### 🔹 Baseline Model
- Embedding Layer
- LSTM
- Fully Connected Layer

### 🔹 Improved Model
- Embedding Layer
- **Bidirectional LSTM (BiLSTM)**
- **Attention Mechanism**
- **Dropout (Regularization)**
- Fully Connected Layer

---

## ⚙️ Tech Stack

- Python
- PyTorch
- NumPy

---

## 📂 Project Structure
```
TweetNLP/
│
├── dataset/                # TweetEval dataset
├── baseline_model.py       # Baseline LSTM model
├── improved_model.py       # Improved BiLSTM model
├── train.py                # Training pipeline
├── evaluate.py             # Evaluation metrics
├── utils.py                # Data processing functions
└── README.md
```
## 📊 Analysis

The confusion matrix shows that the model performs best on neutral and positive classes, while negative samples are often misclassified as neutral. This suggests that neutral sentiment acts as a boundary class, making classification more challenging.

The addition of the attention mechanism allows the model to focus on important words in a sentence, improving contextual understanding and leading to a consistent improvement in F1 score.

---

## 🚀 How to Run

### 1. Train the model
```bash
python3 train.py
```

### 2. Evaluate the model
```bash
python3 evaluate.py
```

---

## 🔄 Model Selection

In both `train.py` and `evaluate.py`, change:
```python
MODEL_TYPE = "baseline"
```

or
```python
MODEL_TYPE = "improved"
```

---

## 📊 Results

| Model                  | Accuracy | Avg F1 Score |
|------------------------|----------|--------------|
| Baseline               | ~0.627   | ~0.5909      |
| BiLSTM                 | ~0.628   | ~0.5936      |
| BiLSTM + Attention     | ~0.626   | ~0.5975      |

---

## 📈 Evaluation Metrics

- Precision
- Recall
- F1 Score *(Primary metric)*

---

## 🧪 Key Improvements

- BiLSTM captures context from both directions
- Dropout reduces overfitting
- Improved model achieves better F1 score

---

## ❌ Constraints

- No pretrained models used 
- Same dataset used for fair comparison

---

## ⭐ Conclusion

The project demonstrates that incremental architectural improvements, such as BiLSTM and attention mechanisms, can enhance model performance without relying on pretrained models, while also providing insights into model behavior through detailed evaluation.

---

## 📚 References

- [TweetNLP Paper (Arxiv)](https://arxiv.org/abs/2206.14774)
- [TweetEval Dataset](https://github.com/cardiffnlp/tweeteval)

---

## 👩‍💻 Author

**Kavya Rai**  
IIT Kharagpur