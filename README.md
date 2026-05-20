# Email Spam Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A supervised machine learning project that classifies text messages as **spam** or **ham** using TF-IDF vectorization and Logistic Regression.

## Project Highlights

- Built an end-to-end text classification pipeline in Python.
- Converted raw email text into numerical features with TF-IDF.
- Trained and evaluated a Logistic Regression model.
- Saved the trained model and vectorizer for repeatable predictions.
- Achieved approximately **96% test accuracy** on the sample dataset.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib

## Project Structure

```text
email_classifier/
|-- data/
|   `-- mail_data.csv
|-- models/
|   |-- model.pkl
|   `-- vectorizer.pkl
|-- src/
|   |-- train.py
|   `-- test.py
|-- requirements.txt
`-- README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Aatmanium/email_classifier.git
cd email_classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run an interactive prediction:

```bash
python src/test.py
```

Example:

```text
Enter your email/message: Congratulations, you won a free prize
Spam Mail
```

## Machine Learning Workflow

1. Load and clean the labeled email dataset.
2. Encode labels as spam or ham.
3. Split the dataset into training and test sets.
4. Transform text with TF-IDF vectorization.
5. Train a Logistic Regression classifier.
6. Evaluate model accuracy and save reusable artifacts.

## Future Improvements

- Add precision, recall, F1-score, and confusion matrix reporting.
- Package prediction logic as a small web app or API.
- Compare Logistic Regression with Naive Bayes and linear SVM.
- Add unit tests for training and prediction utilities.

## Author

**Aatmanium**  
Applied AI Student | Machine Learning Enthusiast | Python Developer
