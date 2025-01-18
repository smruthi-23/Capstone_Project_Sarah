# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from imblearn.over_sampling import RandomOverSampler
import os
import stat
import re
import string
import pickle
import joblib

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Load and Preprocess Data ---
def load_and_preprocess_data(file_path):
    """Load and preprocess dataset."""

    # Load the dataset
    data = pd.read_csv('sample30.csv')

    # Exploratory Data Analysis
    print("Data Info:")
    print(data.info())

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nClass Distribution:")
    print(data['user_sentiment'].value_counts())

    data['cleaned_text'] = data['reviews_text'].apply(clean_text)
    return data

# --- Train Random Forest Model ---
def train_sentiment_model(data, folder_path):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = LabelEncoder().fit_transform(data['user_sentiment'])


    # Handle Class Imbalance
    print("Original class distribution:")
    print(pd.Series(y).value_counts())

    # Remove invalid classes
    valid_classes = [0, 1]  # Define valid classes
    mask = np.isin(y, valid_classes)  # Use NumPy's isin function

    # Filter data using the mask
    X = X[mask]
    y = y[mask]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    from collections import Counter
    print(Counter(y))  # Displays the count of each class

    # Check if data is empty
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("No valid samples remaining after filtering.")

    # Oversample with RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Check new class distribution
    print(Counter(y_resampled))


    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    results = {}

    # Train the Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results['Random Forest'] = accuracy
    print(f"Random Forest Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))



    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    # Set permissions for the folder
    os.chmod(folder_path, stat.S_IRWXU)  # Full permissions for the owner
    print(f"Write permissions set for {folder_path}.")
    print(f"'models' folder created or already exists at: {folder_path}")
    file_path_model = os.path.join(folder_path, 'random_forest_model.pkl')
    file_path_vectorizer = os.path.join(folder_path, 'tfidf_vectorizer.pkl')


    # Save the model and vectorizer
    with open(file_path_model, 'wb') as f:
        pickle.dump(rf_model, f)

    print(f"Model saved successfully at: {file_path_model}")


    with open(file_path_vectorizer, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"Vectorizer saved successfully at: {file_path_vectorizer}")  

    return rf_model, vectorizer

# Recommendation System
def save_user_similarity(data, folder_path):
    ratings = pd.DataFrame({
        'user_id': data['reviews_username'],
        'item_id': data['name'],
        'rating': data['reviews_rating']
    })
    ratings = ratings.groupby(['user_id', 'item_id'], as_index=False).agg({'rating': 'mean'})
    user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    user_item_sparse = csr_matrix(user_item_matrix)
    user_similarity = cosine_similarity(user_item_sparse)
    user_recommendation = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    file_path_user_sim = os.path.join(folder_path, 'user_similarity.pkl')

    # Save the user similarity DataFrame
    with open(file_path_user_sim, 'wb') as f:
        joblib.dump(user_recommendation, f, protocol=4)
    
    print(f"User Recommendation file saved successfully at: {file_path_user_sim}") 

    return user_recommendation   

    

    # Display Recommendations for 10 Users with Sentiment Filtering
def generate_recommendations(user_id, data, rf_model, vectorizer, user_recommendation):
    print(f"user_id: {user_id}, data type: {type(data)}, rf_model: {rf_model}")
    #print(f"user_recommendation: {user_recommendation}")
    ratings = pd.DataFrame({
        'user_id': data['reviews_username'],
        'item_id': data['name'],
        'rating': data['reviews_rating']
    })
    ratings = ratings.groupby(['user_id', 'item_id'], as_index=False).agg({'rating': 'mean'})
    #for username in data['reviews_username'].unique()[:10]:
    user_ratings = ratings[ratings['user_id'] == user_id]
    user_items = user_ratings['item_id'].values

    best_recommendation = user_recommendation

    if user_id not in best_recommendation.index:
        print(f"User: {user_id}\nNo recommendations available (user not in similarity matrix).\n")
    
    user_similarity_scores = best_recommendation.loc[user_id]

        
    # Ensure it's a Pandas Series and contains only numeric data
    if not isinstance(user_similarity_scores, pd.Series):
        user_similarity_scores = pd.Series(user_similarity_scores)


    # Ensure only numeric values are considered
    user_similarity_scores = pd.to_numeric(user_similarity_scores, errors='coerce')

    user_similarity_scores = user_similarity_scores.dropna()

        

    #Sort similar users
    similar_users = user_similarity_scores.sort_values(ascending=False).index
    potential_recommendations = ratings[ratings['user_id'].isin(similar_users)]
    potential_recommendations = potential_recommendations[~potential_recommendations['item_id'].isin(user_items)]

    recommended_scores = potential_recommendations.groupby('item_id')['rating'].mean()
    recommended_items = recommended_scores.sort_values(ascending=False).head(20).index.tolist()

    print(f"User: {user_id}\nRecommended Items:")
    for idx, item in enumerate(recommended_items, start=1):
        print(f" {idx}. {item}")
    print('\n')

    recommended_reviews = data[data['name'].isin(recommended_items)][['name', 'reviews_text']]

    if recommended_reviews.empty:
        print(f"User: {user_id}\nNo reviews found for the recommended items.\n")
        

    # Sentiment Analysis Integration
    def predict_sentiment(text, sentiment_model, vectorizer):
        text_features = vectorizer.transform(text)
        predictions = sentiment_model.predict(text_features)
        return predictions

    recommended_reviews['predicted_sentiment'] = predict_sentiment(
        recommended_reviews['reviews_text'].fillna(''),
        sentiment_model=rf_model,
        vectorizer=vectorizer
    )
    #print(recommended_reviews['predicted_sentiment'].value_counts())
    positive_sentiment_counts = recommended_reviews[recommended_reviews['predicted_sentiment'] == 1].groupby('name').size()
    top_5_items = positive_sentiment_counts.sort_values(ascending=False).head(5).index.tolist()

    print(f"User: {user_id}\nFinal Recommended Items (Top 5 based on sentiment):")
    for idx, item in enumerate(top_5_items, start=1):
        print(f" {idx}. {item}")
    print('\n')

    if 'recommended_items' not in user_recommendation.columns:
        user_recommendation['recommended_items'] = None

    # Ensure the column is of type object
    user_recommendation['recommended_items'] = user_recommendation['recommended_items'].astype(object)

    user_recommendation.at[user_id, 'recommended_items'] = top_5_items

    return top_5_items

# --- Main Execution ---
if __name__ == "__main__":
    # File paths
    dataset_path = 'sample30.csv'
    models_folder = 'models'

    # Load and preprocess data
    data = load_and_preprocess_data(dataset_path)

    
    # Train sentiment model
    rf_model, vectorizer = train_sentiment_model(data, models_folder)

    # Save user similarity matrix
    user_recommendation = save_user_similarity(data, models_folder)

    # Example recommendation
    #example_user_id = data['reviews_username'].iloc[0]
    #recommendations = generate_recommendations(example_user_id, data, user_recommendation)
    #print(f"Recommendations for user {example_user_id}: {recommendations}")    

    