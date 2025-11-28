📈 Tech10 Trading Strategy Lab
End-to-End Quant Research Demo • Snowflake + Streamlit + Python

🔗 Live Demo

👉 Click to open the live Streamlit app:
https://tech10tradingapp-xjdz5dzfyuorn9gdhxil96.streamlit.app/

(Public link, no login required)

🚀 Project Overview

This project is a fully functional quantitative trading analytics laboratory, built end-to-end using:

Snowflake

Streamlit Cloud

Python

Pandas / NumPy

Matplotlib

It enables users to:

✔ Load historical Tech10 price data directly from Snowflake
✔ Select date windows (1M, 6M, YTD, 1Y, 5Y, All)
✔ Run MA / MACD / Buy & Hold strategies
✔ Visualize candlestick charts with buy/sell signals
✔ Compare strategy PnL vs benchmark
✔ View automatically generated strategy interpretation
✔ Use a clean, professional UI suitable for live interviews

📊 Features
1. Trading Strategies

MA Crossover

MACD (12/26/9)

Buy & Hold benchmark

All strategies compute:

Buy/Sell signals

Equity curves vs benchmark

Window-normalized PnL

Real-time interpretation of latest signals

2. Candlestick Chart

High-contrast candlestick style

Light transparent signal arrows

Legend guaranteed even with no signals

Optimized for interview readability

3. Interactive UI

Ticker selector

Date range dropdown

Strategy picker

Toggle backtesting PnL

Modern sidebar styling

4. Secure Secret Management

No credentials appear in code or GitHub.

Secrets are loaded from:

.streamlit/secrets.toml      # local development
Streamlit Cloud Secrets      # cloud deployment

🏗 Project Structure
tech10_trading/
│
├── app/
│   └── app_streamlit.py
├── requirements.txt
├── .gitignore
│
├── alerts/           (ignored)
├── data_ingest/      (ignored)
├── strategy/         (ignored)
└── .streamlit/       (ignored)


Only safe, non-confidential files are pushed to GitHub.

🔧 Running Locally
pip install -r requirements.txt
streamlit run app/app_streamlit.py


Requires a valid .streamlit/secrets.toml.

🌐 Deployment

Push repo to GitHub

Deploy via Streamlit Cloud

Set path to app/app_streamlit.py

Add secrets in Streamlit dashboard

App becomes publicly shareable
