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

df['VALIDATION_CRITERIA_TYPE_ENC'] = label_encoder_criteria.fit_transform(df['VALIDATION_CRITERIA_TYPE'])
df['FAILURE_DETAILS_ENC'] = label_encoder_failure.fit_transform(df['FAILURE_DETAILS'])
df['RECOMMENDATION_ENC'] = label_encoder_recommendation.fit_transform(df['RECOMMENDATION'])

# Defining features (X) and target (y)
X = df[['VALIDATION_CRITERIA_TYPE_ENC', 'FAILURE_DETAILS_ENC']]
y = df['RECOMMENDATION_ENC']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(df['RECOMMENDATION_ENC'].unique()))

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
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

# Decode the predicted recommendation to get the original recommendation text
predicted_recommendation_text = label_encoder_recommendation.inverse_transform(predicted_recommendation)
print(f"Recommended Output: {predicted_recommendation_text[0]}")
