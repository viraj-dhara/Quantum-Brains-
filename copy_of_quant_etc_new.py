import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load dataset (Change file path as needed)
df = pd.read_csv("BTC-USD Price Data (June 2010 - November 2024).csv")
df = df.dropna()

def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# Prepare data
X, y, scaler = preprocess_data(df)
X_train, X_test = X[:-100], X[-100:]
y_train, y_test = y[:-100], y[-100:]

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile & Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate Performance
returns = np.diff(y_pred.squeeze()) / y_pred[:-1].squeeze()
sharpe_ratio = np.mean(returns) / np.std(returns)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.legend()
plt.show()
