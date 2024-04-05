import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pickle

data = pd.read_excel("New Data.xlsx")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# define ML models
model = xgb.XGBRegressor(learning_rate=0.8131380699686145, n_estimators=1325, max_depth=6, min_child_weight=9, subsample=0.9833752640112888, colsample_bytree=1.0)


n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

start_time = time.time()
test_scores = []
train_scores = []
fold_index = 0

for train_index, test_index in kf.split(X):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)


    model.fit(X_train_scaled, y_train)

    # Training performance
    y_train_pred = model.predict(X_train_scaled)

    # Testing performance
    y_test_pred = model.predict(X_test_scaled)


    # Error measurements
    r_lcc_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    r_lcc_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    train_scores.append([r_lcc_train, mse_train,rmse_train, mae_train, mape_train])
    test_scores.append([r_lcc_test, mse_test,rmse_test, mae_test, mape_test])




# Save the trained model
with open("xgboost_CS.pkl", "wb") as file:
    pickle.dump(model, file)