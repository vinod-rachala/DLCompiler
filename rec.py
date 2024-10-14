import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.preprocessing import LabelEncoder

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
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['RECOMMENDATION'])

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=20))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=2)

# Make predictions on new data
test_data = ["api.accountVelocity Expected: approvedCount but none found"]
test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=20)

# Predict recommendations
predictions = model.predict(test_padded)
predicted_label = np.argmax(predictions, axis=1)
predicted_recommendation = le.inverse_transform(predicted_label)

print(f"Predicted Recommendation: {predicted_recommendation[0]}")
