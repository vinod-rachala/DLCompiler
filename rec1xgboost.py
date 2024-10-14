from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Initializing TF-IDF Vectorizer
tfidf_vectorizer_criteria = TfidfVectorizer()
tfidf_vectorizer_failure = TfidfVectorizer()

# Fit and transform the text data using TF-IDF
X_criteria_tfidf = tfidf_vectorizer_criteria.fit_transform(df['VALIDATION_CRITERIA_TYPE'])
X_failure_tfidf = tfidf_vectorizer_failure.fit_transform(df['FAILURE_DETAILS'])

# Combine the two TF-IDF matrices into a single feature set
import scipy
X = scipy.sparse.hstack([X_criteria_tfidf, X_failure_tfidf])

# Encode the recommendation target (this remains label encoded)
label_encoder_recommendation = LabelEncoder()
y = label_encoder_recommendation.fit_transform(df['RECOMMENDATION'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Subtract 1 to make the target zero-indexed (if needed by XGBoost)
y_train_zero_indexed = y_train - 1
y_test_zero_indexed = y_test - 1

import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(df['RECOMMENDATION'].unique()))

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(df['RECOMMENDATION'].unique()))

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
