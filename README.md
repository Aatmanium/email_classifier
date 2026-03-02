# Email Spam Classifier (Machine Learning)

A machine learning project that classifies emails/messages as **Spam** or **Ham (Not Spam)** using:

- TF-IDF Vectorization
- Logistic Regression
- Scikit-learn

TF-IDF stands for:
Term Frequency – Inverse Document Frequency.
It is a method used in Natural Language Processing (NLP) to convert text into numbers so that machine learning models can understand it.

---

## Project Overview

This project builds a text classification model that can detect whether an email is spam or not spam.

The model is trained on labeled email data and achieves:

- **Training Accuracy:** ~96.7%
- **Test Accuracy:** ~96.0%

---

### Machine Learning Concepts
- Text Vectorization (TF-IDF)
- Supervised Learning
- Logistic Regression
- Train/Test Split
- Model Evaluation (Accuracy)
---

## Project Structure
email_classifier
│
├── data/
│ └── mail_data.csv
│
├── models/
│ ├── model.pkl
│ └── vectorizer.pkl
│
├── src/
│ ├── train.py
│ └── test.py
│
├── requirements.txt
└── README.md


---

## 🚀 Quick Start

### 1. Clone the Repository

git clone https://github.com/Aatmanium/email_classifier.git
cd email_classifier



### 2. Install Dependencies

pip install -r requirements.txt



### 3. Train the Model

python src/train.py

This will train the model and save it inside the `models/` folder.



### 4. Test the Model

python src/test.py

Enter any email message when prompted to check if it is Spam or Ham.

---
### Example:
Enter your email/message: Lol your always so convincing.
<br> --> Ham Mail ✅
