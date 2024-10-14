import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from sklearn.model_selection import train_test_split

# Step 1: Load CSV Data
input_file = 'input_data.csv'
data = pd.read_csv(input_file)

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Step 2: Define LSTM Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_seq_length))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 3: Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the tokenizer and model if needed for future use
model.save("lstm_model.h5")

# Step 4: Function to Generate Recommendations for New Data
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

# Step 5: Generate Recommendations for the Whole Dataset
data['GENERATED_RECOMMENDATION'] = data.apply(lambda row: generate_recommendation(row['VALIDATION_CRITERIA_TYPE'], row['FAILURE_DETAILS']), axis=1)

# Step 6: Save Generated Recommendations to CSV
output_file = 'output_recommendations.csv'
data.to_csv(output_file, index=False)

print(f"Generated recommendations have been saved to {output_file}")
