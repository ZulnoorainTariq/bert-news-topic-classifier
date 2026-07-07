# 📰 News Topic Classifier using BERT

A fine-tuned BERT model that classifies news headlines into four categories: **World, Sports, Business, and Sci/Tech**. Built as part of an AI/ML Engineering Internship (Advanced Task 1).

## 🎯 Objective

Fine-tune a transformer model (BERT) to automatically classify news headlines into topic categories, and deploy it as an interactive web application.

## 📊 Dataset

**AG News Dataset** (via Hugging Face Datasets: `fancyzhx/ag_news`)
- 120,000 training samples / 7,600 test samples (full dataset)
- A balanced subset of 3,000 training and 600 test samples was used for fine-tuning
- 4 categories: World, Sports, Business, Sci/Tech

## 🛠️ Methodology

1. **Preprocessing**: Tokenized headlines using `BertTokenizer` (bert-base-uncased), with padding and truncation to a max length of 128 tokens.
2. **Model**: Fine-tuned `bert-base-uncased` using Hugging Face's `Trainer` API for sequence classification (4 output labels).
3. **Training**: 3 epochs, batch size 8, learning rate 2e-5, trained on an NVIDIA RTX 4050 GPU.
4. **Evaluation**: Measured using accuracy and weighted F1-score on the held-out test set.
5. **Deployment**: Built an interactive Streamlit web app for real-time headline classification with confidence scores and probability breakdown.

## 📈 Results

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|-------|---------------|------------------|----------|----------|
| 1 | 0.417 | 0.376 | 90.3% | 90.3% |
| 2 | 0.199 | 0.386 | **90.7%** | **90.7%** |
| 3 | 0.171 | 0.409 | 90.0% | 90.0% |

Best model (epoch 2) was automatically selected for final deployment.

## 🖥️ Tech Stack

- **Model**: BERT (bert-base-uncased) via Hugging Face Transformers
- **Training**: PyTorch, CUDA 12.1
- **Frontend**: Streamlit
- **Libraries**: transformers, datasets, scikit-learn, pandas, accelerate

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure


├── news_classifier.ipynb    # Full notebook: data loading, training, evaluation
├── app.py                   # Streamlit web app for live predictions
├── requirements.txt         # Dependencies
└── README.md

## 💡 Key Observations

- BERT fine-tuning achieved ~90% accuracy with only 3,000 training examples and 3 epochs, demonstrating the power of transfer learning.
- Slight overfitting was observed after epoch 2 (validation loss increased while training loss kept decreasing), so the best checkpoint was retained.
- The model generalizes well to real-world headlines outside the original dataset
