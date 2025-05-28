import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Function to fetch stock data
def get_stock_data(stock, start="2013-01-01", end="2023-01-01"):
    try:
        data = yf.download(stock, start=start, end=end, progress=False)[["Open", "High", "Low", "Close", "Volume"]]
        if data.empty:
            raise ValueError(f"⚠ Stock '{stock}' not found or insufficient data.")
        return data
    except Exception as e:
        print(f"❌ Error downloading data for {stock}: {e}")
        return None

# Function to fetch gold price data
def get_gold_data(start="2013-01-01", end="2023-01-01"):
    try:
        gold = yf.download("GC=F", start=start, end=end, progress=False)[["Open", "High", "Low", "Close", "Volume"]]
        if gold.empty:
            raise ValueError("⚠ Gold price data not found or insufficient data.")
        return gold
    except Exception as e:
        print(f"❌ Error fetching gold prices: {e}")
        return None

# Function to fetch stock fundamentals
def get_fundamentals(stock):
    try:
        ticker = yf.Ticker(stock)
        info = ticker.info  # Fetch fundamental data
        return {
            "Trailing P/E Ratio": info.get("trailingPE", "N/A"),
            "Forward P/E Ratio": info.get("forwardPE", "N/A"),
            "P/B Ratio": info.get("trailingPB", "N/A"),
            "Industry P/E": info.get("forwardPE", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Sales Growth": info.get("revenueGrowth", "N/A"),
            "Debt to Equity": info.get("debtToEquity", "N/A"),
            "Earnings Per Share (EPS)": info.get("trailingEps", "N/A"),
            "Return on Equity (ROE)": info.get("returnOnEquity", "N/A"),
            "Book Value": info.get("bookValue", "N/A")
        }
    except Exception as e:
        print(f"❌ Error fetching fundamentals for {stock}: {e}")
        return None

# Main function to train models and save scalers and fundamentals
def train_and_save_model(stocks):
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=10*365)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    trained_models = {}
    fundamentals_data = {}

    # Fetch gold price data once for plotting overlay
    gold_data = get_gold_data(start_date, end_date)

    for stock in stocks:
        print(f"\n📈 Training model for {stock}...")

        # Get stock data
        data = get_stock_data(stock, start_date, end_date)
        if data is None or data.empty:
            print(f"⚠ Skipping {stock} due to missing data.")
            continue

        # Calculate moving averages
        data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()
        data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
        data["SMA_100"] = data["Close"].rolling(window=100).mean()

        # Plot historical data with gold overlay using secondary y-axis
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.plot(data.index, data["Close"], label=f"{stock} Price", color="blue")
        ax.plot(data.index, data["EMA_200"], label="200-Day EMA", color="red")
        ax.plot(data.index, data["EMA_50"], label="50-Day EMA", color="orange")
        ax.plot(data.index, data["SMA_100"], label="100-Day SMA", color="green")
        if gold_data is not None:
            ax2 = ax.twinx()
            ax2.plot(gold_data.index, gold_data["Close"], label="Gold Price", color="gold", linestyle="--")
            ax2.set_ylabel("Gold Price")
            # Combine legends from both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper left")
        else:
            ax.legend(loc="upper left")
        plt.title(f"{stock} vs Gold Price (Last 10 Years)")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.grid()
        if not os.path.exists("static/images"):
            os.makedirs("static/images")
        plt.savefig(f"static/images/{stock}_past.png")
        plt.close()

        # Prepare data for LSTM model
        feature_columns = ["Open", "High", "Low", "Volume"]
        x = data[feature_columns].to_numpy()
        y = data["Close"].to_numpy().reshape(-1, 1)

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x_scaled = scaler_x.fit_transform(x)
        y_scaled = scaler_y.fit_transform(y)

        if len(x_scaled) < 200:
            print(f"❌ Not enough data for {stock}. Skipping...")
            continue

        # Split data
        xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
        xtrain = np.expand_dims(xtrain, axis=2)
        xtest = np.expand_dims(xtest, axis=2)

        # Define and train LSTM model
        model = Sequential([
            Input(shape=(xtrain.shape[1], 1)),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            Dense(40),
            Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mean_squared_error')
        model.fit(xtrain, ytrain, batch_size=32, epochs=100, verbose=1)

        # Predict future prices (for next 24 months) using a recursive forecasting approach
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=24, freq="M")
        future_predictions = []
        # Start with the last row's features (in original scale)
        feature_columns = ["Open", "High", "Low", "Volume"]
        current_features = data[feature_columns].iloc[-1].to_numpy().reshape(1, -1)
        # Scale the current features
        future_input = scaler_x.transform(current_features)  # shape (1,4)
        for _ in range(24):
            # Reshape input for model: (1,4,1)
            input_reshaped = np.expand_dims(future_input, axis=2)
            pred_scaled = model.predict(input_reshaped)[0][0]
            # Convert prediction back to original scale
            pred = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
            future_predictions.append(pred)
            # Update future_input recursively:
            # First, revert future_input to original scale
            future_input_original = scaler_x.inverse_transform(future_input)  # shape (1,4)
            # Assume that next period's Open, High, Low become the predicted price,
            # while Volume remains unchanged.
            future_input_original[0, 0:3] = pred
            # Re-scale the updated features
            future_input = scaler_x.transform(future_input_original)
        
        future_prices = np.array(future_predictions)

        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, future_prices, marker="o", linestyle="-", color="green", label="Predicted Price")
        plt.title(f"{stock} - Predicted Prices for Next 2 Years")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.savefig(f"static/images/{stock}_future.png")
        plt.close()

        # Save the model and its scalers for the stock
        trained_models[stock] = {
            "model": model,
            "scaler_x": scaler_x,
            "scaler_y": scaler_y
        }
        # Fetch and save fundamentals
        fundamentals_data[stock] = get_fundamentals(stock)
        print(f"📊 Fundamentals for {stock}: {fundamentals_data[stock]}")

    # Save models and fundamentals into pickle files
    with open("models.pkl", "wb") as f:
        pickle.dump(trained_models, f)
    with open("fundamentals.pkl", "wb") as f:
        pickle.dump(fundamentals_data, f)

    print("✅ Models and fundamentals saved successfully.")

if __name__ == '__main__':
    train_and_save_model(["UNIONBANK.NS","RAMASTEEL.NS","RELIANCE.NS","TATASTEEL.NS","TATAMOTORS"])
