from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "vectorizer.pkl"

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    raise FileNotFoundError("Model files not found. Run `python src/train.py` first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

user_input = input("Enter your email/message: ")
input_features = vectorizer.transform([user_input])
prediction = model.predict(input_features)

if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")
