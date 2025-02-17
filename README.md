# Fake News Detection System

A machine learning model that detects fake news articles with 92.74% accuracy using TF-IDF vectorization and PassiveAggressive Classification.

## Features
- Text preprocessing and feature extraction using TF-IDF
- PassiveAggressive Classifier for binary classification
- 92.74% accuracy on test dataset
- Balanced precision with minimal false positives

## Technologies Used
- Python
- scikit-learn
- pandas
- NumPy

## Setup
1. Clone the repository

2. Install dependencies

pip install numpy pandas scikit-learn


3. Run the script

python fake_news_detector.py


## Model Performance
- Overall Accuracy: 92.74%
- True Negatives (Correctly identified FAKE): 588
- True Positives (Correctly identified REAL): 587
- False Positives: 50
- False Negatives: 42

## Project Structure

fake_news_detector.py - Main script for fake news detection
data/ - Directory containing the dataset
README.md - This file


