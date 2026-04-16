from flask import Flask, render_template, request
import joblib
import re
import string
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load trained artifacts
vectorizer = joblib.load(BASE_DIR / 'vectorizer.jb')
model = joblib.load(BASE_DIR / 'lr_model.jb')


def clean_text(text: str) -> str:
    """Apply the same preprocessing used during training."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    return text


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    prediction_class = None
    news_input = ''

    if request.method == 'POST':
        news_input = request.form.get('news', '').strip()

        if news_input:
            cleaned_input = clean_text(news_input)
            transformed_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_input)[0]

            if prediction == 1:
                prediction_text = 'Real News Detected'
                prediction_class = 'real'
            else:
                prediction_text = 'Fake News Detected'
                prediction_class = 'fake'
        else:
            prediction_text = 'Please enter some news text to analyze.'
            prediction_class = 'warning'

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        prediction_class=prediction_class,
        news_input=news_input
    )


if __name__ == '__main__':
    app.run(debug=True)
