from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
import joblib
from model import (
    load_and_preprocess_data,
    train_sentiment_model,
    save_user_similarity,
    generate_recommendations,
)
# Initialize Flask app
app = Flask(__name__)

# File paths and constants
dataset_path = 'sample30.csv'
models_folder = 'models'

print(f"Current working directory: {os.getcwd()}")
print(f"Dataset exists: {os.path.exists(dataset_path)}")

# Global variables for models and data
data = None
rf_model = None
vectorizer = None
user_recommendation = None

# Initialize components
def initialize_app():
    print("initialize_app called.")  # Debugging
    global data, rf_model, vectorizer, user_recommendation

    # Ensure models folder exists
    os.makedirs(models_folder, exist_ok=True)

    try:
        data = load_and_preprocess_data(dataset_path)
        print("Data loaded successfully!")
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}. Please ensure the file exists.")
        data = None 
        return

    # Train sentiment model
    print("Training sentiment model...")
    if not os.path.exists(os.path.join(models_folder, 'random_forest_model.pkl')):
        rf_model, vectorizer = train_sentiment_model(data, models_folder)

    # Save user similarity matrix
    print("Saving user similarity matrix...")
    if not os.path.exists(os.path.join(models_folder, 'user_similarity.pkl')):
        user_recommendation = save_user_similarity(data, models_folder)

    print("Initialization complete!")


# Load models
#model_folder = os.path.join(os.getcwd(), 'models')
try:
    with open(os.path.join(models_folder, 'random_forest_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    with open(os.path.join(models_folder, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(models_folder, 'user_similarity.pkl'), 'rb') as f:
        user_recommendation = joblib.load(f)
    
    print("Models and similarity matrix loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Initializing components...")

    initialize_app()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html', error=None)

@app.route('/recommend', methods=['POST','GET'])
def recommend():
    global data
    print(f"Debug: Data type in /recommend: {type(data)}")  # Debugging

    if data is None:
        try:
            print("Reinitializing data...")
            data = load_and_preprocess_data(dataset_path)
        except FileNotFoundError:
            return render_template('index.html', recommendations=None, user=None, error="Dataset not loaded. Please check the server configuration.")
    
    if request.method == 'POST':
        user_id = request.form.get('user_id')  # Get the user ID from the form input
    else:
        user_id = request.args.get('user_id')  # For GET requests

    #print(f"Received user_id: {user_id}")  # Debugging

    if not user_id:
        return render_template('index.html', recommendations=None, error="Please provide a valid user_id")

    top_5_items = generate_recommendations(user_id, data, rf_model, vectorizer, user_recommendation)

    #print(f"Top 5 items for {user_id}: {top_5_items}")  # Debugging
    # Handle errors if recommendations are not found
    if not top_5_items:
        return render_template('index.html', recommendations=None, error="No recommendations available.")

    # Pass the top 5 items to the front-end
    return render_template('index.html', recommendations=top_5_items, user=user_id, error=None)

if __name__ == '__main__':
    print("Initializing application...")
    initialize_app() 
    print("Starting Flask app...")
    app.run(debug=True)