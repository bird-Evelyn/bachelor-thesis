import pandas as pd
import joblib

X_pred = pd.read_csv("D:/桌面/毕设/2023国赛C/pred_single.csv")
# 载入模型
xgb_model = joblib.load("D:/桌面/毕设/2023国赛C/model/xgboost/xgboost_best_model0.pkl")
# 模型预测
y_pred = xgb_model.predict(X_pred)

# 将预测结果添加到原数据中
X_pred['prediction'] = y_pred

# 确保 'code' 列存在并且是正确的列名
# 根据 'code' 分组并将结果重新整形
result_df = X_pred.groupby('3')['prediction'].apply(list).reset_index()

# 拆分列表为多列
result_df = pd.DataFrame(result_df['prediction'].tolist(), index=result_df.index)

# 设置结果 DataFrame 的列名
result_df.columns = [f'day_{i+1}' for i in range(result_df.shape[1])]

# 将 'code' 列添加回结果
result_df['3'] = X_pred['3'].unique()

# 调整列的顺序，使 'code' 在最前面
result_df = result_df[['3'] + [f'day_{i+1}' for i in range(result_df.shape[1] - 1)]]

# 打印结果
print(result_df)
result_df.to_csv("D:/桌面/毕设/2023国赛C/prediction_results.csv", index=False)

# print(y_pred)