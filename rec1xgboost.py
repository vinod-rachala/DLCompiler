import pandas as pd

# Sample dataset
data = {
    'VALIDATION_CRITERIA_TYPE': [
        'api.accountVelocity', 'api.paymentDetails', 'api.accountVelocity', 'api.accountBalance'
    ],
    'FAILURE_DETAILS': [
        'Expected : approvedCount but none found',
        'Expected : paymentAmount but none found',
        'Expected : approvedCount but none found',
        'Expected : balanceAmount but none found'
    ],
    'RECOMMENDATION': [
        'Expected : PurchaseAmount but none found',
        'Expected : finalPaymentAmount but none found',
        'Expected : PurchaseAmount but none found',
        'Expected : transactionAmount but none found'
    ]
}

df = pd.DataFrame(data)
print(df)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Label encoding for the categorical columns
label_encoder_criteria = LabelEncoder()
label_encoder_failure = LabelEncoder()
label_encoder_recommendation = LabelEncoder()

# Encode the validation criteria, failure details, and recommendations
df['VALIDATION_CRITERIA_TYPE_ENC'] = label_encoder_criteria.fit_transform(df['VALIDATION_CRITERIA_TYPE'])
df['FAILURE_DETAILS_ENC'] = label_encoder_failure.fit_transform(df['FAILURE_DETAILS'])
df['RECOMMENDATION_ENC'] = label_encoder_recommendation.fit_transform(df['RECOMMENDATION'])

# Defining features (X) and target (y)
X = df[['VALIDATION_CRITERIA_TYPE_ENC', 'FAILURE_DETAILS_ENC']]
y = df['RECOMMENDATION_ENC']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Adjust y_train and y_test to be zero-indexed
y_train_zero_indexed = y_train - 1
y_test_zero_indexed = y_test - 1
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(df['RECOMMENDATION_ENC'].unique()))

# Train the model
model.fit(X_train, y_train_zero_indexed)

# Predict on test data
y_pred = model.predict(X_test)

# Add 1 back to the predictions to match the original labels
y_pred_original_labels = y_pred + 1

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_original_labels)
print(f"Model Accuracy: {accuracy}")


# Example new test case
new_test_case = pd.DataFrame({
    'VALIDATION_CRITERIA_TYPE': ['api.accountVelocity'],
    'FAILURE_DETAILS': ['Expected : approvedCount but none found']
})

# Encode the new test case
new_test_case['VALIDATION_CRITERIA_TYPE_ENC'] = label_encoder_criteria.transform(new_test_case['VALIDATION_CRITERIA_TYPE'])
new_test_case['FAILURE_DETAILS_ENC'] = label_encoder_failure.transform(new_test_case['FAILURE_DETAILS'])

# Predict recommendation
new_X = new_test_case[['VALIDATION_CRITERIA_TYPE_ENC', 'FAILURE_DETAILS_ENC']]
predicted_recommendation = model.predict(new_X)

# Add 1 back to match the original class labels
predicted_recommendation_original = predicted_recommendation + 1

# Decode the predicted recommendation to get the original recommendation text
predicted_recommendation_text = label_encoder_recommendation.inverse_transform(predicted_recommendation_original)
print(f"Recommended Output: {predicted_recommendation_text[0]}")
