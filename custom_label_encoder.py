import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Sample Data
data = {
    'VALIDATION_CRITERIA_TYPE': [
        "api.accountVelocity", "api.purchaseAmount", "api.accountVelocity",
        "api.accountBalance", "api.purchaseAmount"
    ],
    'FAILURE_DETAILS': [
        "Expected : approvedCount but none found", 
        "Expected : balance but none found", 
        "Expected : approvedCount but none found", 
        "Expected : balance but none found",
        "Expected : approvedCount but none found"
    ],
    'RECOMMENDATION': [
        "Expected : PurchaseAmount but none found", 
        "Expected : accountBalance but none found",
        "Expected : PurchaseAmount but none found",
        "Expected : accountBalance but none found",
        "Expected : PurchaseAmount but none found"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the features and target
X = df[['VALIDATION_CRITERIA_TYPE', 'FAILURE_DETAILS']]
y = df['RECOMMENDATION']

# Combine 'VALIDATION_CRITERIA_TYPE' and 'FAILURE_DETAILS' for text processing
X_combined = X['VALIDATION_CRITERIA_TYPE'] + " " + X['FAILURE_DETAILS']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Use TF-IDF for feature extraction
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Ensure the target variable is in numerical format
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train XGBoost Classifier
model = XGBClassifier()
model.fit(X_train_tfidf, y_train_encoded)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Handle unknown labels in the prediction
def handle_unknown_labels(pred_labels, label_encoder):
    """Map unknown labels to a default recommendation."""
    default_label = "<UNKNOWN RECOMMENDATION>"
    mapped_labels = []
    for label in pred_labels:
        # Try to inverse transform the label
        try:
            mapped_label = label_encoder.inverse_transform([label])[0]
        except ValueError:
            # Handle unknown label
            mapped_label = default_label
        mapped_labels.append(mapped_label)
    return mapped_labels

# Remap predicted labels back to original labels or unknown label
y_pred_labels = handle_unknown_labels(y_pred, label_encoder)

# Output accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Output the recommendations
print("Predicted Recommendations:")
for true_val, pred_val in zip(y_test, y_pred_labels):
    print(f"True: {true_val} | Predicted: {pred_val}")

# Example prediction with unseen data
new_data = ["api.accountVelocity Expected : approvedCount but none found"]
new_data_tfidf = vectorizer.transform(new_data)
new_pred = model.predict(new_data_tfidf)
new_pred_label = handle_unknown_labels(new_pred, label_encoder)
print(f"\nNew Data Prediction: {new_pred_label[0]}")
