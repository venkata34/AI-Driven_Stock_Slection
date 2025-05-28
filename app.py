import matplotlib
matplotlib.use("Agg")

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
# Initialize Flask app
app = Flask(__name__)

# Ensure static folder exists for saving graphs
if not os.path.exists("static/images"):
    os.makedirs("static/images")

# Load trained models and fundamentals
def load_models():
    try:
        with open("models.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def load_fundamentals():
    try:
        with open("fundamentals.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

models_data = load_models()
fundamentals = load_fundamentals()

# Function to get stock data
def get_stock_data(stock, start, end):
    try:
        data = yf.download(stock, start=start, end=end, progress=False)[["Open", "High", "Low", "Close", "Volume"]]
        if data.empty:
            return None
        # Compute technical indicators
        data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
        data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()
        return data
    except Exception as e:
        print(f"Error fetching data for {stock}: {e}")
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock = request.form['stock']
        if stock not in models_data:
            return jsonify({"error": f"No trained model found for {stock}."}), 400

        model_info = models_data[stock]
        model = model_info["model"]
        scaler_x = model_info["scaler_x"]
        scaler_y = model_info["scaler_y"]

        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")

        data = get_stock_data(stock, start_date, today)
        if data is None:
            return jsonify({"error": f"Stock '{stock}' not found."}), 404

        # Preprocessing latest features for prediction
        feature_columns = ["Open", "High", "Low", "Volume"]
        latest_features = scaler_x.transform(data[feature_columns].iloc[-1].to_numpy().reshape(1, -1))
        latest_features = np.expand_dims(latest_features, axis=2)

        # Predict the next close price
        predicted_price_scaled = model.predict(latest_features)[0][0]
        predicted_price = scaler_y.inverse_transform(np.array([[predicted_price_scaled]]))[0][0]

        # Warnings based on fundamentals
        stock_fundamentals = fundamentals.get(stock, {})
        warnings = []
        if stock_fundamentals.get("Market Cap", 0) < 5000000000:
            warnings.append("⚠️ Small-cap stock!")
        if stock_fundamentals.get("Trailing P/E Ratio", 0) > stock_fundamentals.get("Industry P/E", 1):
            warnings.append("⚠️ High P/E ratio!")
        if stock_fundamentals.get("Sales Growth", 0) < 0:
            warnings.append("⚠️ Declining sales growth!")

        # Generate past performance graph with gold overlay
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.plot(data.index, data['Close'], label="Stock Price", color="blue")
        ax.plot(data.index, data['EMA_50'], label="50-Day EMA", color="green")
        ax.plot(data.index, data['EMA_200'], label="200-Day EMA", color="red")
        # For past graph, try to overlay gold if available
        gold_data = yf.download("GC=F", start=start_date, end=today, progress=False)[["Close"]]
        if gold_data is not None and not gold_data.empty:
            ax2 = ax.twinx()
            ax2.plot(gold_data.index, gold_data["Close"], label="Gold Price", color="gold", linestyle="--")
            ax2.set_ylabel("Gold Price")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper left")
        else:
            ax.legend(loc="upper left")
        plt.title(f"{stock} - Past 10 Years with Indicators & Gold")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.grid()
        plt.savefig(f"static/images/{stock}_past.png")
        plt.close()

        # Predict next 2 years (24 months) using recursive forecasting
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=24, freq="M")
        future_predictions = []
        # Initialize with the last row's features (original scale)
        current_features = data[feature_columns].iloc[-1].to_numpy().reshape(1, -1)
        future_input = scaler_x.transform(current_features)
        for _ in range(24):
            input_reshaped = np.expand_dims(future_input, axis=2)
            pred_scaled = model.predict(input_reshaped)[0][0]
            pred = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
            future_predictions.append(pred)
            # Update input recursively: assume predicted close becomes the next open, high, low; volume remains the same
            future_input_original = scaler_x.inverse_transform(future_input)
            future_input_original[0, 0:3] = pred
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

        return jsonify({
            "stock": stock,
            "predicted_close_price": round(float(predicted_price), 2),
            "warnings": warnings,
            "fundamentals": stock_fundamentals,
            "past_graph": f"/static/images/{stock}_past.png",
            "future_graph": f"/static/images/{stock}_future.png"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
