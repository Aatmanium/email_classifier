from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "mail_data.csv"
MODEL_DIR = ROOT_DIR / "models"

raw_mail_data = pd.read_csv(DATA_PATH)
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), "")

mail_data.loc[mail_data["Category"] == "spam", "Category"] = 0
mail_data.loc[mail_data["Category"] == "ham", "Category"] = 1

X = mail_data["Message"]
y = mail_data["Category"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_features, y_train)

train_pred = model.predict(X_train_features)
test_pred = model.predict(X_test_features)

print("Accuracy on training data =", accuracy_score(y_train, train_pred))
print("Accuracy on test data     =", accuracy_score(y_test, test_pred))

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")

print("Model and vectorizer saved in models folder.")
