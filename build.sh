#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Optional: Add any additional build steps here if needed
# For example, if you need to train your model during build:
# python save_model.py 