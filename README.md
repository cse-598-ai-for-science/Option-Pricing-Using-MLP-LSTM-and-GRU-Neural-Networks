train_from_yf.py

This repository contains a helper script `train_from_yf.py` that pulls option chain data from Yahoo Finance
(via the included `data_loader.py`) and trains a small MLP on the fetched data.

Quick start

1. Create a python virtual environment and activate it (recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Run a quick smoke run (example):

   python train_from_yf.py --ticker AAPL --expirations 2 --epochs 5 --batch_size 16

Notes
- `data_loader.py` contains `yfinance_to_model_df` which maps Yahoo Finance option chain fields to the
  features expected by the model.
- If you don't want to compute GARCH-based conditional volatility, omit `--use_garch`.
- The script trains on the log of C/S by default; change `prepare_xy` if you prefer raw prices.

