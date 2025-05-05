import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify
import joblib
import os

# === Load and Clean Data ===
df = pd.read_csv("fake_and_real_news.csv")

# Drop missing Text or label
df.dropna(subset=['Text', 'label'], inplace=True)

# Clean label values: remove blank, strip spaces, convert to uppercase
df = df[df['label'].astype(str).str.strip() != '']
df['label'] = df['label'].astype(str).str.upper().str.strip()

# Keep only valid labels
df = df[df['label'].isin(['FAKE', 'REAL'])]

# Convert labels to binary
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# === Features and Labels ===
X = df['Text']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
print("=== Evaluation Report ===")
print(classification_report(y_test, model.predict(X_test)))

# Save model and vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# === Flask App ===
app = Flask(__name__)
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "Fake News Detection API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing \"text\" field'}), 400
    input_text = data['text']
    transformed = vectorizer.transform([input_text])
    prediction = model.predict(transformed)[0]
    result = "REAL" if prediction == 1 else "FAKE"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
