from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Model files not found. Please run the model saving code first.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        # Clean and vectorize the title
        cleaned_title = clean_text(title)
        title_vector = vectorizer.transform([cleaned_title])
        
        # Predict
        prediction = model.predict(title_vector)[0]
        
        # Get probability
        proba = np.max(model.predict_proba(title_vector)) * 100
        
        result = {
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': f'{proba:.2f}%'
        }
        
        return render_template('index.html', title=title, result=result)

if __name__ == '__main__':
    app.run(debug=True) 