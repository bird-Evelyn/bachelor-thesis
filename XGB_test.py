import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)

def rse_scorer(true, pred):
    mse = mean_squared_error(true, pred)
    return mse / np.sum(np.square(true - np.mean(true)))

def rae_scorer(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.sum(np.abs(true - pred))
    squared_error_den = np.sum(np.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss

# 读取数据
train_data = pd.read_csv("D:/桌面/毕设/2023国赛C/数据/all/train_selected_ml_day.csv")
val_data = pd.read_csv("D:/桌面/毕设/2023国赛C/数据/all/val_selected_ml_day.csv")
test_data = pd.read_csv("D:/桌面/毕设/2023国赛C/数据/all/test_selected_ml_day.csv")
X_train = pd.concat([train_data.drop(columns=['total_qty']), val_data.drop(columns=['total_qty'])])
y_train = pd.concat([train_data['total_qty'], val_data['total_qty']])
X_test, y_test = test_data.drop(columns=['total_qty']), test_data['total_qty']

# 载入模型
xgb_model = joblib.load("D:/桌面/毕设/2023国赛C/model/xgboost/xgboost_best_model0.pkl")

# 模型预测
y_pred = xgb_model.predict(X_test)

# 计算并打印评估指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rse = rse_scorer(y_test, y_pred)
rae = rae_scorer(y_test, y_pred)

print(f"MAE:  {mae:.4f}, MSE:  {mse:.4f}, MAPE: {mape:.4f}, RSE:  {rse:.4f}, RAE:  {rae:.4f}")
