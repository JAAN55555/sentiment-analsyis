from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# 🔥 Emotion detection model (better than sentiment)
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    result = classifier(text)[0]

    label = result["label"]
    confidence = round(result["score"] * 100, 2)

    return render_template("index.html",
                           prediction=label,
                           confidence=confidence,
                           text=text)

if __name__ == "__main__":
    app.run(debug=True)
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)