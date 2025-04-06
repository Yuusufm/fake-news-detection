import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define possible data file paths
possible_paths = [
    'data/news.csv',
    'news.csv',
    'fake_news.csv',
    'data/fake_news.csv'
]

# Find the correct data file
data_file = None
for path in possible_paths:
    if os.path.exists(path):
        data_file = path
        break

if data_file is None:
    # Create a simple sample dataset if no data file is found
    print("No data file found. Creating a sample dataset.")
    data = {
        'title': [
            "Trump says he will win election by landslide",
            "Scientists discover new species in Amazon",
            "Aliens spotted in New York City",
            "Unemployment rate drops to historic low",
            "Diet pills guarantee 20 pound weight loss in a week",
            "Global warming study finds conclusive evidence",
            "New miracle cure for all cancers discovered",
            "Stock market hits all-time high"
        ],
        'label': ['FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE', 'REAL']
    }
    df = pd.DataFrame(data)
    print("Created sample dataset with 8 examples")
else:
    # Load the dataset
    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Shape of dataset: {df.shape}")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply text cleaning to title
print("Cleaning text...")
df['cleaned_title'] = df['title'].apply(clean_text)

# Split data
print("Preparing data for training...")
X = df['cleaned_title']
y = (df['label'] == 'FAKE').astype(int)  # Convert to binary (0=REAL, 1=FAKE)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(solver='liblinear')
model.fit(X_train_vec, y_train)

# Save the model and vectorizer
print("Saving model and vectorizer...")
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved successfully!")

# Optional: Test accuracy
y_pred = model.predict(X_test_vec)
accuracy = np.mean(y_pred == y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%") 