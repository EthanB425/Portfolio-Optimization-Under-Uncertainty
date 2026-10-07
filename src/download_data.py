import yfinance as yf
import pandas as pd
import os

TICKERS = ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "VNQ", "DBC"]

START = "2007-01-01"
END = "2026-10-01"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading ETF prices...")
prices = yf.download(TICKERS, start=START, end=END, auto_adjust=True)["Close"]
prices = prices[TICKERS]  # keep a consistent column order
prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
print(f"  Saved {prices.shape[0]} rows x {prices.shape[1]} columns to data/prices.csv")

print("  First valid date per ticker:")
print(prices.apply(lambda col: col.first_valid_index()).to_string())

print("Downloading risk-free rate (13-week T-bill yield, ^IRX)...")
irx = yf.download("^IRX", start=START, end=END, auto_adjust=False)["Close"]
irx.to_csv(os.path.join(DATA_DIR, "irx.csv"))
print(f"  Saved {len(irx)} rows to data/irx.csv")

print("Done.")