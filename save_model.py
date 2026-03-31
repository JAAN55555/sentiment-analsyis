import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 🔹 Load dataset
df = pd.read_csv("customer_reviews.csv", low_memory=False)

# 🔹 Remove empty values
df = df.dropna(subset=["text", "Emotion"])

# 🔹 Convert to string (important fix)
df["text"] = df["text"].astype(str)
df["Emotion"] = df["Emotion"].astype(str)

# 🔹 Features and labels
X = df["text"]
y = df["Emotion"]

# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Improved Vectorizer (better accuracy)
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # bigrams
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔹 Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 🔹 Accuracy
preds = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, preds)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# 🔹 Save model + vectorizer
joblib.dump((model, vectorizer), "sentiment_model.pkl")

print("Model saved successfully!")