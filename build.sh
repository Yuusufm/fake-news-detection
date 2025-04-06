#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Make the save_model.py script run during build
echo "Training and saving model during build..."
python save_model.py

# Optional: Add any additional build steps here if needed
# For example, if you need to train your model during build:
# python save_model.py 