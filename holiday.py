import pandas as pd
import chinese_calendar

# 读取文件
file_path = 'data.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 将第一列转换为时间格式
data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
# 添加一列，判断第一列的日期是否为节假日
data['is_holiday'] = data.iloc[:, 0].apply(lambda x: 1 if chinese_calendar.is_holiday(x.date()) else 0)

# 保存结果到新文件
data.to_csv('data.csv', index=False)  
