# 📊 Real-Time Stock Market Prediction

This project is a real-time stock price prediction tool built using **Machine Learning (XGBoost)** and **yfinance**. It fetches historical stock data, trains a regression model, and predicts **future closing prices** for top stocks like **Apple, Google, Tesla, Amazon**, and more (**25 tickers in total**).


## Features

- ✅ Real-time stock data fetching using `yfinance`  
- 📅 Time-series lag feature engineering  
- 🤖 Model training using `XGBoost Regressor`  
- 📈 Forecasts future prices (30+ days)  
- 📊 Performance metrics: **MSE**, **R² Score**  
- 📉 Interactive plots: Actual vs Predicted + Future forecast  
- 🧠 Predict multiple stocks by user input (e.g., `AAPL`, `MSFT`, etc.)


## Tech Stack

- Python  
- yfinance  
- XGBoost  
- scikit-learn  
- matplotlib  
- pandas, numpy


## 🚀 After executing

You'll be prompted to enter one or more stock tickers (e.g., AAPL,MSFT) and the number of days to forecast (minimum 30).

📌 Sample Output
📊 Actual vs Predicted stock price graph

📈 Forecast chart for future N days

📋 Metrics like Mean Squared Error and R²

📄 Tabular data of Open, High, Low, Close, Volume for the last 30 days

📚 Learning Outcomes
Time-series data preprocessing

Real-world data handling via APIs

Performance tuning of regression models

Visual storytelling with stock data

Practical understanding of stock price behavior

🔗 Connect
If you're interested in stock forecasting, finance + ML, or building projects like this — feel free to fork the repo, suggest improvements, or connect with me on LinkedIn ( https://www.linkedin.com/in/avatapalli-tejaswini/ ) .
