import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/stocks/AAPL/open_close.csv', parse_dates=True, index_col='Date')
# plt.plot(df['Adj Close'].values)
# plt.show()

# df['Year'] = pd.DatetimeIndex(df['Date']).year
# df['Month'] = pd.DatetimeIndex(df['Date']).month
# df['Day'] = pd.DatetimeIndex(df['Date']).day
#
# df = df.drop('Date', axis=1)

# df = pd.get_dummies(data=df, columns=['Year', 'Month', 'Day'], drop_first=True)
df['daily_pct_change'] = df['Adj Close'].pct_change()
df['daily_pct_change'].fillna(0, inplace=True)

df['daily_log_returns'] = np.log(df['Adj Close'].pct_change()+1)
df['daily_log_returns'].fillna(0, inplace=True)

df['cum_daily_return'] = (1 + df['daily_pct_change']).cumprod()
df['cum_daily_return'].fillna(0, inplace=True)

df['moving_avg_10'] = df['Adj Close'].rolling(window=10).mean()
df['moving_avg_40'] = df['Adj Close'].rolling(window=40).mean()
df['moving_avg_252'] = df['Adj Close'].rolling(window=252).mean()

print(df.head())
# train = df.drop(['Close', 'Adj Close'], axis=1)
# test = df['Adj Close']
# plt.plot(df['Adj Close'])
# plt.show()