from flask import Flask, render_template, request
import os
import joblib

app = Flask(__name__)

# Load your saved model
model, vectorizer = joblib.load("sentiment_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]

        # Convert text to vector
        text_vec = vectorizer.transform([text])

        # Predict sentiment
        prediction = model.predict(text_vec)[0]

        # Simple confidence
        confidence = 100

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            text=text
        )

    return render_template("index.html")


# IMPORTANT for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
