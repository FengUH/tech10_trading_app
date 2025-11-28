# 📈 Tech10 Trading Strategy Lab  
End-to-End • Snowflake + Streamlit + Python

---

## 🔗 Live Demo

👉 **Click to open the live Streamlit app:**  
https://tech10tradingapp-xjdz5dzfyuorn9gdhxil96.streamlit.app/

*(Public link — no login required)*

---

## 🚀 Project Overview

This project is a fully functional quantitative trading analytics laboratory, built end-to-end with:

- **Snowflake**
- **Streamlit Cloud**
- **Python**
- **Pandas / NumPy**
- **Matplotlib**

It enables users to:

- ✔ Load historical Tech10 price data from Snowflake  
- ✔ Select time windows (1M, 6M, YTD, 1Y, 5Y, All)  
- ✔ Run **MA**, **MACD**, or **Buy & Hold** strategies  
- ✔ Visualize **candlestick charts with Buy/Sell signals**  
- ✔ Compare **strategy PnL vs benchmark**  
- ✔ View **automatically generated strategy interpretation**  
- ✔ Use a **clean, interview-ready professional UI**

---

## 📊 Features

### **1. Trading Strategies**
- **MA Crossover**
- **MACD (12/26/9)**
- **Buy & Hold Benchmark**

All strategies compute:

- Buy/Sell signals  
- Equity curves vs benchmark  
- Window-normalized PnL  
- Human-readable interpretation of the latest signal  

---

### **2. Candlestick Chart**
- High-contrast professional candlestick styling  
- Transparent green/red signal arrows  
- Buy/Sell legend appears even when no trades occur  
- Optimized for interview readability  

---

### **3. Interactive UI**
- Ticker selector  
- Date-range dropdown  
- Strategy picker  
- Toggle backtesting PnL  
- Clean modern sidebar styling  

---

### **4. Secure Secret Management**

No credentials appear in any code or GitHub repository.

Secrets are loaded from:

- `.streamlit/secrets.toml` — **local development**  
- **Streamlit Cloud Secrets** — cloud deployment  

Only non-sensitive files are pushed to GitHub.

---

## 🏗 Project Structure

```text
tech10_trading/
│
├── app/
│   └── app_streamlit.py
│
├── requirements.txt
├── .gitignore
│
├── alerts/          (ignored)
├── data_ingest/     (ignored)
├── strategy/        (ignored)
└── .streamlit/      (ignored)

All sensitive or internal scripts remain local and are excluded via `.gitignore`.

---

## 🔧 Running Locally

```bash
pip install -r requirements.txt
streamlit run app/app_streamlit.py

Requires a valid .streamlit/secrets.toml file.
