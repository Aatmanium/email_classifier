import os
import joblib

# Paths
MODEL_PATH = os.path.join("models", "model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Take input from user
user_input = input("Enter your email/message: ")

# Convert text to features
input_features = vectorizer.transform([user_input])

# Predict
prediction = model.predict(input_features)

# Output result
if prediction[0] == 1:
    print("Ham Mail ✅")
else:
    print("Spam Mail 🚨")