import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, make_scorer)

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
train_data = pd.read_csv("../../数据/all/train_selected_ml_day.csv")
val_data = pd.read_csv("../../数据/all/val_selected_ml_day.csv")
X_train = pd.concat([train_data.drop(columns=['total_qty']), val_data.drop(columns=['total_qty'])])
y_train = pd.concat([train_data['total_qty'], val_data['total_qty']])

# 定义超参数搜索空间
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500,1000,1500],
    'subsample': [0.7, 0.8, 0.9, 1.0],   #训练实例的子抽样比例
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_lambda':[0,0.1,1,10],
    'max_bin': [256,384,512]             #连续特征转为离散特征，最大使用的离散箱数
}

# 定义模型
xgb_model = xgb.XGBRegressor(
    booster='gbtree',
    device='cpu',
    n_jobs=10,
    validate_parameters=True,
    objective='reg:squarederror',
    random_state=1234,
    tree_method='hist',
    sampling_method='uniform',          #训练实例采样方法
    grow_policy='lossguide',
    enable_categorical = True
)

# 使用网格搜索
grid_search = GridSearchCV(
    estimator = xgb_model,
    param_grid = param_grid,
    cv=5,  # 5折交叉验证
    scoring={
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'RAE': make_scorer(rae_scorer, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'RSE': make_scorer(rse_scorer, greater_is_better=False),
        'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    },
    refit='MSE',  # 以 MSE 作为最终模型选择标准
    verbose=2  # 显示搜索进度
)

# 训练模型
grid_search.fit(X_train, y_train)

# 获取最优超参数
best_params = grid_search.best_params_
print("最佳超参数:", best_params)

# 保存模型
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'xgboost_best_model0.pkl')
print("模型已保存为 xgboost_best_model0.pkl")

# 将搜索结果转换为 DataFrame 并保存
df_results = pd.DataFrame(grid_search.cv_results_)
df_results.to_csv("hyperparameter_results_day_0.csv", index=False)
