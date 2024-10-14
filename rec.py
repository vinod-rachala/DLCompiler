import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'VALIDATION_CRITERIA_TYPE': ['Type1', 'Type2', 'Type1', 'Type3', 'Type2', 'Type1', 'Type3'],
    'FAILURE_DETAILS': ['Error in module A', 'Timeout in module B', 'Crash in module C', 'Error in module D', 
                        'Timeout in module A', 'Crash in module E', 'Error in module F'],
    'RECOMMENDATION': ['Restart A', 'Increase timeout B', 'Debug module C', 'Restart D', 
                       'Increase timeout A', 'Debug module E', 'Restart F']
})

# Combine validation type and failure details as input text
data['INPUT_TEXT'] = data['VALIDATION_CRITERIA_TYPE'] + " " + data['FAILURE_DETAILS']

# Prepare input (X) and output (y) sequences
X = data['INPUT_TEXT']
y = data['RECOMMENDATION']

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X + y)

# Convert text to sequences
X_seq = tokenizer.texts_to_sequences(X)
y_seq = tokenizer.texts_to_sequences(y)

# Pad sequences to ensure equal length for input and output
max_seq_length = max(max(len(seq) for seq in X_seq), max(len(seq) for seq in y_seq))
X_padded = pad_sequences(X_seq, maxlen=max_seq_length, padding='post')
y_padded = pad_sequences(y_seq, maxlen=max_seq_length, padding='post')

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed

# Define the model architecture
model = Sequential()

# Embedding layer to convert words into vector representations
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_seq_length))

# LSTM layer
model.add(LSTM(256, return_sequences=True))

# Dense layer for generating recommendations
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))



def generate_recommendation(validation_criteria, failure_details):
    # Combine the input into a single string
    input_text = validation_criteria + " " + failure_details
    
    # Convert input text to sequence
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq_padded = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')

    # Predict the recommendation
    prediction = model.predict(input_seq_padded)

    # Convert prediction to text
    predicted_seq = np.argmax(prediction, axis=-1)
    predicted_text = tokenizer.sequences_to_texts(predicted_seq)[0]

    return predicted_text

# Example of generating recommendations
validation_criteria = "Type1"
failure_details = "Timeout in module C"
recommendation = generate_recommendation(validation_criteria, failure_details)
print(f"Generated Recommendation: {recommendation}")
