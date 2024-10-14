import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

# Step 1: Generate Synthetic Data

validation_criteria_types = [
    "api.accountVelocity", "api.transactionLimit", "api.purchaseLimit", 
    "api.userAccess", "api.paymentThreshold", "api.fraudCheck"
]

failure_details_templates = [
    "Expected: {} but none found", 
    "Expected: {} but null found", 
    "Expected: {} but mismatched value",
    "Expected: {} but zero count"
]

expected_fields = ["approvedCount", "transactionID", "PurchaseAmount", "userLimit", "thresholdValue"]

recommendations_templates = [
    "Expected: PurchaseAmount but none found", 
    "Expected: transactionLimit but null found", 
    "Expected: PurchaseAmount but zero count"
]

# Create 2000 synthetic rows
num_rows = 2000
synthetic_data = {
    "VALIDATION_CRITERIA_TYPE": [],
    "FAILURE_DETAILS": [],
    "RECOMMENDATION": []
}

for _ in range(num_rows):
    validation_type = random.choice(validation_criteria_types)
    expected_field = random.choice(expected_fields)
    failure_detail = random.choice(failure_details_templates).format(expected_field)
    recommendation = random.choice(recommendations_templates)

    synthetic_data["VALIDATION_CRITERIA_TYPE"].append(validation_type)
    synthetic_data["FAILURE_DETAILS"].append(failure_detail)
    synthetic_data["RECOMMENDATION"].append(recommendation)

# Convert to DataFrame
df = pd.DataFrame(synthetic_data)

# Step 2: Preprocess the Data

# Tokenizer setup
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['VALIDATION_CRITERIA_TYPE'] + " " + df['FAILURE_DETAILS'])

# Convert text to sequences
X = tokenizer.texts_to_sequences(df['VALIDATION_CRITERIA_TYPE'] + " " + df['FAILURE_DETAILS'])

# Pad sequences to ensure uniform input length
X = pad_sequences(X, padding='post', maxlen=20)

# Encode the recommendations into numerical labels
le = LabelEncoder()
y = le.fit_transform(df['RECOMMENDATION'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the LSTM Model

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=20))  # Increase embedding dimension
model.add(LSTM(128, return_sequences=False))  # Increase LSTM units to capture complexity
model.add(Dropout(0.5))  # Add dropout to avoid overfitting
model.add(Dense(64, activation='relu'))  # Increase dense layer units
model.add(Dropout(0.5))  # Another dropout for regularization
model.add(Dense(len(le.classes_), activation='softmax'))

# Configure the optimizer with a custom learning rate
learning_rate = 0.001  # Custom learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the optimizer and metrics
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Step 4: Train the Model

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 5: Evaluate the Model's Accuracy

accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {accuracy[1] * 100:.2f}%")

# Step 6: Make Sample Predictions

# Example test data
test_data = ["api.accountVelocity Expected: approvedCount but none found"]
test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=20)

# Predict recommendations
predictions = model.predict(test_padded)
predicted_label = np.argmax(predictions, axis=1)
predicted_recommendation = le.inverse_transform(predicted_label)

print(f"Sample Prediction: {predicted_recommendation[0]}")
