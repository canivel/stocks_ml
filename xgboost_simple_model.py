import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


df = pd.read_csv('data/stocks/AAPL/open_close.csv', parse_dates=True, index_col='Date')

df['daily_pct_change'] = df['Adj Close'].pct_change()
df['daily_pct_change'].fillna(0, inplace=True)

df['daily_log_returns'] = np.log(df['Adj Close'].pct_change()+1)
df['daily_log_returns'].fillna(0, inplace=True)

df['cum_daily_return'] = (1 + df['daily_pct_change']).cumprod()
df['cum_daily_return'].fillna(0, inplace=True)

df['moving_avg_10'] = df['Adj Close'].rolling(window=10).mean()
df['moving_avg_40'] = df['Adj Close'].rolling(window=40).mean()
df['moving_avg_252'] = df['Adj Close'].rolling(window=252).mean()

# plt.plot(df['Adj Close'].values)
# plt.show()

# df['Year'] = pd.DatetimeIndex(df['Date']).year
# df['Month'] = pd.DatetimeIndex(df['Date']).month
# df['Day'] = pd.DatetimeIndex(df['Date']).day
#
# df = df.drop('Date', axis=1)

# df = pd.get_dummies(data=df, columns=['Year', 'Month', 'Day'], drop_first=True)

# print(df.head())
# # train = df.drop(['Close', 'Adj Close'], axis=1)
# # test = df['Adj Close']
# plt.plot(df['Adj Close'])
# plt.show()

def visualize_features_importance(xgb_train):
    xgb.plot_importance(xgb_train)
    plt.show()

def plot_xgb_trees(xg_reg, num_tree=0):
    # Plot the first tree
    xgb.plot_tree(xg_reg, num_trees=num_tree)
    plt.show()
    # Plot the last tree sideways
    # xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
    # plt.show()

def predict(X, DM_test, params, DM_train, watchlist):
    xg_reg = xgb.train(params, DM_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)
    preds = xg_reg.predict(DM_test)

    # rmse = np.sqrt(mean_squared_error(y_valid, preds))
    # print("RMSE {}".format(rmse))
    return preds

if __name__ == "__main__":
    is_train = True
    search_params = False
    X = df.drop(['Close', 'Adj Close'], axis=1).values

    y = df['Adj Close'].values
    print(X.shape, y.shape)

    params = {}
    params['eta'] = 0.001
    params['booster'] = 'gblinear'
    params['objective'] = 'reg:linear'
    params['eval_metric'] = 'rmse'
    params['max_depth'] = 10
    params['silent'] = 0
    params['lambda'] = 100
    params['num_boost_round'] = 1000
    params['colsample_bytree'] = 0.65
    params['subsample'] = 0.8

    if search_params:
        pipeline_steps = [("st_scaler",StandardScaler()), ("xgb_model", xgb.XGBRegressor(params=params))]
        xgb_pipeline = Pipeline(pipeline_steps)

        gbm_param_grid = {'xgb_model__subsample': np.arange(.05, 1, .05),
                          'xgb_model__max_depth': np.arange(3, 20, 1),
                          'xgb_model__colsample_bytree': np.arange(.1, 1.05, .05)}

        randomized_neg_rmse = RandomizedSearchCV(estimator=xgb_pipeline,
                                                param_distributions=gbm_param_grid,
                                                n_iter=10,
                                                scoring='neg_mean_squared_error',
                                                cv=4)

        randomized_neg_rmse.fit(X, y)

        print("Best rmse: ", np.abs(randomized_neg_rmse.best_score_))

        print("Best model: ", randomized_neg_rmse.best_estimator_)

    if is_train == True:
        split_test = 100
        split = 4000
        X_train, y_train, X_valid, y_valid = X[:split], y[:split], X[split:-split_test], y[split:-split_test]
        X_test, y_test = X[-split_test:], y[-split_test:]
        print(X.shape, X_train.shape, X_valid.shape, X_test.shape)

        DM_train = xgb.DMatrix(data=X_train, label=y_train)
        DM_valid = xgb.DMatrix(data=X_valid, label=y_valid)
        DM_test = xgb.DMatrix(data=X_test, label=y_test)
        watchlist = [(DM_train, 'train'), (DM_valid, 'valid')]

        xg_reg = xgb.train(params, DM_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=10)
        preds = xg_reg.predict(DM_test)

        #visualize_features_importance(DM_train)
        plt.plot(preds)
        plt.plot(y_test)
        plt.legend(['predict close', 'real close'], loc='best')
        plt.show()