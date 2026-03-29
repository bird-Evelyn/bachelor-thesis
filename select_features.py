# 读取数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
#from matplotlib.font_manager import font_manager.FontProperties  # 导入FontProperties



# 哪里需要显示中文就在哪里设置
data_path = r'D:/桌面/毕设/2023国赛C/数据/all/all_data.csv'
data = pd.read_csv(data_path)

# 去除第一列时间数据
#data = data .drop(data.columns[[0, 1, 7]], axis=1)

# 将因变量与自变量分开
X = data.drop(columns=['total_qty'])
y = data['total_qty']
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1234)
# 训练模型
feature_selector = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
feature_selector.fit(X_train, y_train)
# 输出特征重要性并将其从大到小排序
importances = feature_selector.feature_importances_
indices=np.argsort(importances)[::-1]
feat_labels=X.columns
for f in range(X_train.shape[1]):
    print ("%2d) %-*s %f" %(f+1,30,feat_labels[indices[f]],importances[indices[f]]))

plt.style.use('ggplot')
font = font_manager.FontProperties(family='SimHei')
plt.title("Feature importance——all", fontproperties=font)
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.legend(prop=font)
plt.show()
plt.savefig("数据/大类/1.辣椒.png")

#选择特征/评估模型
sfm = SelectFromModel(feature_selector, threshold=0.01, prefit=True)
# 将数据转换为重要性大于0.01的特征，删除小于0.01的特征
# 一步完成转换和拼接
train_data = pd.concat([
    pd.DataFrame(sfm.transform(X_train)).reset_index(drop=True),
    pd.DataFrame(y_train).reset_index(drop=True)
], axis=1)
val_data = pd.concat([
    pd.DataFrame(sfm.transform(X_val)).reset_index(drop=True),
    pd.DataFrame(y_val).reset_index(drop=True)
], axis=1)
test_data = pd.concat([
    pd.DataFrame(sfm.transform(X_test)).reset_index(drop=True),
    pd.DataFrame(y_test).reset_index(drop=True)
], axis=1)
train_data.to_csv("数据/all/train_selected_ml_day.csv", index=False)
val_data.to_csv("数据/all/val_selected_ml_day.csv", index=False)
test_data.to_csv("数据/all/test_selected_ml_day.csv", index=False)


