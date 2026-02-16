import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data_path = os.path.join("data", "mail_data.csv")
raw_mail_data = pd.read_csv(data_path)

# Replace null values
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), "")

# Label encoding
mail_data.loc[mail_data["Category"] == "spam", "Category"] = 0
mail_data.loc[mail_data["Category"] == "ham", "Category"] = 1

X = mail_data["Message"]
Y = mail_data["Category"].astype(int)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=5
)

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluate
train_pred = model.predict(X_train_features)
test_pred = model.predict(X_test_features)

print("Accuracy on training data =", accuracy_score(Y_train, train_pred))
print("Accuracy on test data     =", accuracy_score(Y_test, test_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved in models folder.")
