# Quantexa

Quantexa is a Python-based quantitative analysis and backtesting project designed to work with market data and perform financial analysis.

## 📌 Project Overview

This project provides a backend system for fetching market data, performing quantitative analysis, and running backtesting operations.

## ✨ Features

- Market data fetching
- Quantitative analysis
- Backtesting
- Performance analysis
- Data validation
- Automated testing

## 🛠️ Technologies Used

- Python
- FastAPI
- Pandas
- NumPy
- Pytest

## 📂 Project Structure

```text
Quantexa
│
├── app
│   ├── services
│   │   ├── backtester.py
│   │   ├── data_fetcher.py
│   │   └── quant_engine.py
│   │
│   ├── tests
│   │   ├── test_backtest.py
│   │   └── test_quant.py
│   │
│   ├── main.py
│   └── schemas.py
│
├── requirements.txt
├── run.py
└── .gitignore
