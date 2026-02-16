# Spam-Ham Mail Classifier (Logistic Regression)

This project classifies emails/messages as **Spam** or **Ham** using **TF-IDF** + **Logistic Regression**.

## Project Structure
- `data/` dataset (`mail_data.csv`)
- `src/train.py` train + evaluate + save model
- `src/predict.py` load saved model and predict message
- `models/` saved model and vectorizer

## Setup
```bash
pip install -r requirements.txt
