from flask import Flask, render_template, request
import pandas as pd
import re
import string
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_train():
    fake_path = BASE_DIR / "Fake_small.csv"
    true_path = BASE_DIR / "True_small.csv"

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # If dataset has title and text columns
    if "title" in df.columns and "text" in df.columns:
        df["content"] = (
            df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
        )
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("").astype(str)
    else:
        raise ValueError("CSV files must contain either 'text' or both 'title' and 'text' columns.")

    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    return vectorizer, model

# Train model once when app starts
vectorizer, model = load_and_train()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    prediction_class = None
    news_input = ""

    if request.method == "POST":
        news_input = request.form.get("news", "").strip()

        if news_input:
            cleaned_input = clean_text(news_input)
            transformed_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_input)[0]

            if prediction == 1:
                prediction_text = "Real News Detected"
                prediction_class = "real"
            else:
                prediction_text = "Fake News Detected"
                prediction_class = "fake"
        else:
            prediction_text = "Please enter some news text to analyze."
            prediction_class = "warning"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        prediction_class=prediction_class,
        news_input=news_input
    )

if __name__ == "_main_":
    app.run(debug=True)