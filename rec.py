import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Sample Data
data = {
    "VALIDATION_CRITERIA_TYPE": ["api.accountVelocity", "api.transactionLimit", "api.accountVelocity"],
    "FAILURE_DETAILS": ["Expected: approvedCount but none found", 
                        "Expected: transactionID but null found", 
                        "Expected: approvedCount but none found"],
    "RECOMMENDATION": ["Expected: PurchaseAmount but none found", 
                       "Expected: transactionLimit but null found", 
                       "Expected: PurchaseAmount but none found"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

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

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=20))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, verbose=0)

# Evaluate the model's accuracy
accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {accuracy[1] * 100:.2f}%")

# Make predictions on new data
test_data = ["api.accountVelocity Expected: approvedCount but none found"]
test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=20)

# Predict recommendations
predictions = model.predict(test_padded)
predicted_label = np.argmax(predictions, axis=1)
predicted_recommendation = le.inverse_transform(predicted_label)

print(f"Sample Prediction: {predicted_recommendation[0]}")
