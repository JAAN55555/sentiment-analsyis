from flask import Flask, render_template, request
import os
from transformers import pipeline

app = Flask(__name__)

# Load model once
classifier = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]

        result = classifier(text)[0]
        label = result["label"]
        confidence = round(result["score"] * 100, 2)

        return render_template(
            "index.html",
            prediction=label,
            confidence=confidence,
            text=text
        )

    return render_template("index.html")


# IMPORTANT: Only ONE run block
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)