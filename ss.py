import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import os

# Set stock symbol and data range
STOCK = "AAPL"        # You can change to TSLA, MSFT, etc.
PERIOD = "60d"
INTERVAL = "1h"

# Download historical data
def fetch_data():
    df = yf.download(STOCK, period=PERIOD, interval=INTERVAL, progress=False)
    df.dropna(inplace=True)
    return df

# Prepare data for ML model
def prepare_data(df):
    df["timestamp"] = np.arange(len(df))
    X = df[["timestamp"]]
    y = df["Close"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Setup matplotlib
plt.style.use('ggplot')  # avoid seaborn issues
fig, ax = plt.subplots()

def animate(i):
    df = fetch_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)

    df["Predicted"] = model.predict(df[["timestamp"]])

    # Clear previous plot
    ax.clear()

    # Plot real and predicted prices
    ax.plot(df.index, df["Close"], label='Real Price', color='blue')
    ax.plot(df.index, df["Predicted"], label='Predicted Price', color='orange', linestyle='--')

    ax.set_title(f'{STOCK} Stock Price Prediction (Updated: {datetime.datetime.now().strftime("%H:%M:%S")})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)

    # Clear console and print current data table (last 5 rows)
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
    print(f"📊 {STOCK} Stock Table - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(df[["Open", "High", "Low", "Close", "Predicted", "Volume"]].tail(5))

# Animate every 10 seconds
ani = FuncAnimation(fig, animate, interval=10000)

plt.tight_layout()
plt.show()
