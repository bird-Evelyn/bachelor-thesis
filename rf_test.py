import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, make_scorer
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

# 定义模型
rf = RandomForestRegressor(n_estimators=1000, criterion='poisson', ccp_alpha=0, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)

# 保存模型
joblib.dump(rf, "rf_month_1.pkl")#pickle这个包可以做序列化对象的保存

# 模型预测
y_pred = rf.predict(X_test)

# 计算并打印评估指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rse = rse_scorer(y_test, y_pred)
rae = rae_scorer(y_test, y_pred)

print(f"MAE:  {mae:.4f}, MSE:  {mse:.4f}, MAPE: {mape:.4f}, RSE:  {rse:.4f}, RAE:  {rae:.4f}")


# font = font_manager.FontProperties(family='SimHei')
# def plot_true_vs_pred(y_test, y_pred):
#     plt.figure(figsize=(8,6))
#     plt.plot(y_test, label='True', marker='o')
#     plt.plot( y_pred, label='Predicted', marker='x')
#     plt.legend()
#     plt.xlabel('Sample')
#     plt.ylabel('Value')
#     plt.title('True vs Predicted_辣椒_RF',fontproperties=font)
#     plt.grid()
#     plt.show()
#
# # 用法
# plot_true_vs_pred(y_test, y_pred)