import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the dataset
data = '''(Paste your CSV text here)'''  # Replace with the full CSV text
df = pd.read_csv(pd.compat.StringIO(data))

# --- Prepare Data ---
# Select features (excluding 'country' as it's categorical)
features = ['tourism_receipts', 'tourism_arrivals', 'tourism_exports', 
            'tourism_expenditures', 'gdp', 'inflation']

# Sort by year and country to create a consistent sequence
df_sorted = df.sort_values(['year', 'country'])[features]

# Handle missing values by filling with the mean (or drop if preferred)
df_sorted = df_sorted.fillna(df_sorted.mean())

# Check if enough data exists
if len(df_sorted) < 6:
    print("Error: Insufficient data. Need at least 6 rows for seq_length=5.")
    exit()

# Create sequences from the entire dataset
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Sequence of features
        y.append(data[i + seq_length, 0])  # Next tourism_receipts value
    return np.array(X), np.array(y)

seq_length = 5  # Number of timesteps to look back
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_sorted)
X_lstm, y_lstm = create_sequences(scaled_data, seq_length)

# Split into train and test sets
train_size = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, len(features))))
model.add(Dense(1))  # Output layer for tourism_receipts
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, 
          validation_split=0.1, verbose=1)

# Evaluate the model
y_pred_lstm = model.predict(X_test_lstm)

# Inverse transform predictions and actual values
y_test_lstm_inv = scaler.inverse_transform(
    np.concatenate((y_test_lstm.reshape(-1, 1), 
    np.zeros((len(y_test_lstm), len(features)-1))), axis=1))[:, 0]
y_pred_lstm_inv = scaler.inverse_transform(
    np.concatenate((y_pred_lstm, 
    np.zeros((len(y_pred_lstm), len(features)-1))), axis=1))[:, 0]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_lstm_inv, y_pred_lstm_inv))
print(f"LSTM RMSE (All Data): {rmse}")

# Display results
results = pd.DataFrame({'Actual': y_test_lstm_inv, 'Predicted': y_pred_lstm_inv})
print("Sample Predictions:")
print(results.head())

# Optional: Map predictions back to country-year for interpretation
test_indices = range(train_size + seq_length, len(df_sorted))
test_df = df_sorted.iloc[test_indices]
print("\nPredictions with Context (Country, Year):")
print(test_df[['tourism_receipts']].reset_index(drop=True).join(results))