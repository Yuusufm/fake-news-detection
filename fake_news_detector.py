# Import required libraries
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# read the data
df = pd.read_csv('./data/news.csv')
print(f"Shape of dataset: {df.shape}")
print("\nFirst few records:")
print(df.head())


labels = df.label
print("\nFirst few labels:")
print(labels.head())

# split and test subsets
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], 
    labels, 
    test_size=0.2, 
    random_state=7
)


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# Convert text to TF-IDF features
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)


# Train the classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {round(score*100,2)}%')

# confusion matrix 
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))