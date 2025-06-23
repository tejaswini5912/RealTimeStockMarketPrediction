import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import datetime

# Common stock ticker names
TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon", "TSLA": "Tesla", "META": "Meta",
    "NVDA": "NVIDIA", "JPM": "JPMorgan", "V": "Visa", "JNJ": "Johnson & Johnson", "WMT": "Walmart",
    "PG": "Procter & Gamble", "DIS": "Disney", "MA": "Mastercard", "HD": "Home Depot", "BAC": "Bank of America",
    "INTC": "Intel", "KO": "Coca-Cola", "NFLX": "Netflix", "ADBE": "Adobe", "PFE": "Pfizer", "CSCO": "Cisco",
    "PEP": "Pepsi", "XOM": "Exxon Mobil", "MRK": "Merck"
}

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data.dropna()

def create_lag_features(data, lags=5):
    for i in range(1, lags + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    return data.dropna()

def train_predict_future(data, lags=5, future_days=30):
    data = create_lag_features(data, lags)
    X = data[[f'lag_{i}' for i in range(1, lags + 1)]]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Future predictions
    last_known = X.iloc[-1].values
    future_preds = []
    for _ in range(future_days):
        next_pred = model.predict([last_known])[0]
        future_preds.append(next_pred)
        last_known = np.roll(last_known, shift=1)
        last_known[0] = next_pred

    return y_test, y_pred, future_preds, mse, r2, data

def plot_results(ticker, y_test, y_pred, future_preds, mse, r2, data):
    plt.figure(figsize=(14, 7))
    test_dates = y_test.index
    plt.plot(test_dates, y_test, label='Actual Prices', color='royalblue', marker='o')
    plt.plot(test_dates, y_pred, label='Predicted Prices', color='darkorange', linestyle='--', marker='x')

    plt.axvline(x=test_dates[0], color='green', linestyle='--', label='Train-Test Split')
    plt.annotate(f'Max Predicted: {max(y_pred):.2f}', 
                 xy=(test_dates[np.argmax(y_pred)], max(y_pred)),
                 xytext=(test_dates[np.argmax(y_pred)], max(y_pred) + 5),
                 arrowprops=dict(facecolor='red'), color='brown')
    plt.annotate(f'Min Predicted: {min(y_pred):.2f}',
                 xy=(test_dates[np.argmin(y_pred)], min(y_pred)),
                 xytext=(test_dates[np.argmin(y_pred)], min(y_pred) - 5),
                 arrowprops=dict(facecolor='green'), color='green')

    plt.title(f"{ticker} Stock Price Prediction\nOptimized MSE: {mse:.2f} | R²: {r2:.2f}", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot future forecast
    future_dates = pd.date_range(start=datetime.datetime.today(), periods=len(future_preds))
    plt.figure(figsize=(14, 6))
    plt.plot(future_dates, future_preds, label='Future Predictions', marker='o', linestyle='-')
    plt.title(f"{ticker} - Future {len(future_preds)} Days Forecast")
    plt.xlabel("Future Dates")
    plt.ylabel("Predicted Close Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print dataset info
    print(f"\n--- {ticker} Stock Dataset (last 30 entries) ---")
    print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(30).to_string())

def main():
    tickers_input = input(f"Enter stock tickers (comma-separated, e.g., AAPL,MSFT): ").upper().split(',')
    tickers = [t.strip() for t in tickers_input if t.strip() in TICKER_NAMES]

    if not tickers:
        print("No valid tickers provided.")
        return

    future_days = int(input("Enter number of future days to predict (min 30): "))
    if future_days < 30:
        print("Minimum is 30 days. Setting to 30.")
        future_days = 30

    start_date = "2024-01-01"
    end_date = "2025-06-02"

    for ticker in tickers:
        print(f"\nProcessing {ticker} ({TICKER_NAMES[ticker]})...")
        try:
            data = fetch_stock_data(ticker, start_date, end_date)
            y_test, y_pred, future_preds, mse, r2, full_data = train_predict_future(data, future_days=future_days)
            plot_results(ticker, y_test, y_pred, future_preds, mse, r2, full_data)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()