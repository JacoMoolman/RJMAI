import yfinance as yf
data = yf.download("EURUSD=X", start='2022-12-01', end='2022-12-31')
print(data)

