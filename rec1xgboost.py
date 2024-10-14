import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Read data from CSV
def read_csv_data(file_path):
    # Assuming the CSV contains columns: VALIDATION_CRITERIA_TYPE, FAILURE_DETAILS, RECOMMENDATION
    df = pd.read_csv(file_path)
    return df

# Step 2: Preprocess the data using TF-IDF
def preprocess_data(df):
    tfidf = TfidfVectorizer(max_features=1000)  # Reduced to 1000 most frequent terms
    X = tfidf.fit_transform(df['VALIDATION_CRITERIA_TYPE'] + " " + df['FAILURE_DETAILS'])
    
    # Encode the recommendations into numerical labels
    le = LabelEncoder()
    y = le.fit_transform(df['RECOMMENDATION'])
    
    return X, y, tfidf, le

# Step 3: Train the XGBoost model with optimized parameters
def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, learning_rate=0.1, subsample=0.5)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the XGBoost Model
def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Make Sample Predictions
def predict_sample(model, tfidf, le):
    test_data = ["api.accountVelocity Expected: approvedCount but none found"]
    test_vector = tfidf.transform(test_data)
    prediction = model.predict(test_vector)
    predicted_recommendation = le.inverse_transform(prediction)
    
    print(f"Sample Prediction: {predicted_recommendation[0]}")

# Main function to run the entire process
if __name__ == "__main__":
    # Read CSV data
    file_path = 'data.csv'  # Change this to the actual path of your CSV file
    df = read_csv_data(file_path)
    
    # Preprocess data
    X, y, tfidf, le = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = train_xgboost(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Make a sample prediction
    predict_sample(model, tfidf, le)
