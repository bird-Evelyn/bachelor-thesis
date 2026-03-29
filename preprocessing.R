# 导入包永远放在文件最前面，下载包不放在文件中，除非要分享给完全不会用R语言的人
library(tidyverse)
library(readxl)
library(proxy)
# 不使用123等后缀进行命名，所有变量名都必须有实际含义，
# 能够仅从变量名就知道它是什么东西，不同单词缩写或不同含义之间用"_"进行连接
# 赋值语句必须用 "<-"，函数传参用"="，两边必须各有一个空格
# 所有运算符的两边都必须有空格，除非是"!"，即非运算符
single_info <- read_excel("附件1.xlsx")
data <- read_excel("附件2.xlsx")
single_cost <- read_excel("附件3.xlsx")

## 1) 整合数据
# 按照层次，#的个数表示层级，文字必须与#之间空一格
# 有层次地写，多利用转行，避免超过每一行字符数的限制
# 列名转换为英文是一个好习惯
# 底下要对日月年进行分析，所以别把日期放在变量里，对日期提取日月年作为三个变量
# 可以根据习惯利用relocate变换列的位置
data <- data %>%  
  rename("日期" = "销售日期") %>%
  left_join(single_info %>% 
              select(单品编码, 分类编码, 单品名称, 分类名称), 
            by = "单品编码") %>%
  left_join(single_cost %>% 
              select(单品编码, 日期, `批发价格(元/千克)`), 
            by = c("单品编码", "日期")) %>%  
  filter(销售类型 != "退货") %>% 
  rename(all_date = 日期, code = 单品编码, qty = `销量(千克)`, 
         price = `销售单价(元/千克)`, on_sale = 是否打折销售, 
         code_name = 单品名称, classify_code = 分类编码, 
         classify_name = 分类名称, cost = `批发价格(元/千克)`) %>% 
  mutate(all_date = ymd(all_date), year = year(all_date), 
         month = month(all_date), date = day(all_date)) %>% 
  select(-销售类型)

# 步骤划分尽量简洁明了，不要划分太多步，有些相似的步骤合并起来
# 别编号missing_summary1234，直接用对应的来写变量名
# 数据检查放在前面，处理放在后面，按照逻辑顺序来
## 2) 数据检查与处理
### 1 缺失值检查(无缺失值)
missing_summary <- data %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "missing_count") %>%
  filter(missing_count > 0)
### 2 NAN检查与处理（无缺失值NAN）
nan_summary <- data %>%
  summarise(across(everything(), ~sum(is.nan(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "missing_count") %>%
  filter(missing_count > 0)
### 3 重复值检查与处理(无重复值)
# 直接distinct去除重复值就好了，也别数行数，看右边数据行数是否发生变化就行了
data <- data %>% 
  distinct(.keep_all = TRUE)
### 4 异常值处理
#### a 除去非打折情况下，price < cost的情况
data <- data %>% 
  filter(!(on_sale == "否" & price < cost))
#### b 在数据处理过程中，发现了一条数据的qty值为160，在3σ之外了，于是删除
data <- data %>% 
  filter(!(qty==160))
### 5 查看在同一天中同一产品的成本和售价是否一样
n_price_cost <- data %>%  
  group_by(code, all_date) %>%  
  summarise(  
    unique_price_count = n_distinct(price),  
    unique_cost_count = n_distinct(cost), 
    .drop = TRUE
  )
#### a 成本是不变的，但是部分产品的售价在一天中是变换的
#### 针对同一天中产品price变换的情况下，只保留price出现次数最多的数据
price_change_delete <- n_price_cost %>% 
  filter(unique_price_count != 1) %>% 
  inner_join(data, by = c("code" = "code", "all_date" = "all_date")) %>% 
  group_by(code, all_date, price) %>% 
  summarise(count = n(), .groups = 'drop') %>% 
  arrange(code, all_date, desc(count)) %>% 
  group_by(code, all_date) %>% 
  slice(-1) %>%
  ungroup()
data <- data %>% 
  anti_join(price_change_delete, by = c("code", "all_date", "price"))
### 6 将分类变量转换为因子类型，并将 on_sale 转换为 0/1 (是==1，否==0)
# 用fct转换因子类型，而不是baser中的as.factor
# 批量转换因子类型，用across函数
# 因子并非必须转化的，看分析过程决定是否要转化为因子，比如说需要得到它的所有取值
# 这种情况下转化为因子然后以后每次取提取因子水平肯定是最快的
# 或者需要对分类数据的水平进行操作的时候，很需要转化为因子数据
data <- data %>%  
  mutate(  
    on_sale = ifelse(on_sale == "是", 1, 0),   
    across(c(classify_name, classify_code, code_name, code, 
             on_sale, year, month, date), as_factor),
    across(c(classify_code, code), as.numeric)
  )  
### 7 去除不必要的列
data <- data %>%
  select(-c(扫码销售时间))

## 3) 提取特征
data <- data %>%  
  mutate(  
    # 季度特征
    quarter = case_when(  
      month %in% c(1, 2, 3) ~ 0,  
      month %in% c(4, 5, 6) ~ 1,  
      month %in% c(7, 8, 9) ~ 2,  
      month %in% c(10, 11, 12) ~ 3  
    ), 
    # 周末
    is_weekend = ifelse(wday(all_date) %in% c(6, 7), 0, 1),  
    # 旬
    phase = case_when(  
      date %in% c(1:10) ~ 0,
      date %in% c(11:20) ~ 1,
      date %in% c(21:31) ~ 2
      )
  )
# 保存结果，转python写节假日特征
write.csv(data, "data.csv", row.names = FALSE)
# 读取添加完节假日特征后的数据
data <- read_csv("data.csv")
# 改变顺序
data <- data %>%
  mutate(holiday_or_weekend = ifelse(is_holiday == 0 | is_weekend == 0, 0, 1)) %>%
  relocate(all_date, year, month, date, classify_name, classify_code, 
           code_name, code, cost, price, qty, on_sale, quarter, 
           phase, is_weekend, is_holiday, holiday_or_weekend)


save_classify <- function(data, classify) {
  filtered_data <- data %>%
    filter(classify_code == classify)
  
  # 1. 日
  sum_day <- filtered_data %>%
    group_by(all_date) %>%
    summarise(
      total_qty = sum(qty, na.rm = TRUE),
      total_cost = weighted.mean(cost, qty),
      total_price = weighted.mean(price, qty),
      total_on_sale = mean(on_sale),
      .groups = "drop")

  result_day <- left_join(
    filtered_data %>% 
      select(-c(classify_name, classify_code, code_name, code, cost, price, qty, on_sale)) %>% 
      distinct(), 
    sum_day, by = c("all_date")) %>% 
    select(-all_date) %>% 
    relocate(total_qty, .after = last_col())

  write.csv(result_day, 
            file = paste0("classify_", classify, "_dataset_day.csv"),
            row.names = FALSE)
  
  # # 2. 周
  # sum_week <- filtered_data %>% 
  #   mutate(
  #     week = isoweek(all_date),
  #     year = isoyear(all_date)
  #   ) %>% 
  #   group_by(year, week) %>% 
  #   summarise(
  #     total_qty = sum(qty, na.rm = TRUE),
  #     total_cost = weighted.mean(cost, qty),
  #     total_price = weighted.mean(price, qty),
  #     total_on_sale = mean(on_sale),
  #     .groups = "drop")
  # 
  # result_week <- left_join(
  #   filtered_data %>% 
  #     mutate(week = isoweek(all_date)) %>% 
  #     select(-c(all_date, date, classify_name, classify_code, 
  #               code_name, code, cost, price, qty, on_sale, phase, 
  #               is_weekend, is_holiday, holiday_or_weekend)) %>% 
  #     distinct(), 
  #   sum_week, by = c("year", "week")) %>% 
  #   relocate(total_qty, .after = last_col())
  # 
  # write.csv(result_week, 
  #           file = paste0("classify_", classify, "_dataset_week.csv"),
  #           row.names = FALSE)
  
  # 3. 月
  sum_month <- filtered_data %>%
    group_by(year, month) %>%
    summarise(
      total_qty = sum(qty, na.rm = TRUE),
      total_cost = weighted.mean(cost, qty),
      total_price = weighted.mean(price, qty),
      total_on_sale = mean(on_sale),
      .groups = "drop")
  
  result_month <- left_join(
    filtered_data %>% 
      select(-c(all_date, date, classify_name, classify_code, 
                code_name, code, cost, price, qty, on_sale,
                phase, is_weekend, is_holiday, holiday_or_weekend)) %>% 
      distinct(), 
    sum_month, by = c("year", "month")) %>% 
    relocate(total_qty, .after = last_col())
  
  write.csv(result_month, 
            file = paste0("classify_", classify, "_dataset_month.csv"),
            row.names = FALSE)
}

lapply(1:6, function(x) save_classify(data, x))


# 确定每个大类未来几日的特征
## 不同种类预测特征值滑动计算(利用前7天的平均)
data_list <- list(
  data_1 = read.csv("D:/桌面/毕设/2023国赛C/数据/大类/1.辣椒/classify_1_dataset_day.csv"),
  data_2 = read.csv("D:/桌面/毕设/2023国赛C/数据/大类/2.花叶/classify_2_dataset_day.csv"),
  data_3 = read.csv("D:/桌面/毕设/2023国赛C/数据/大类/3.水生根/classify_3_dataset_day.csv"),
  data_4 = read.csv("D:/桌面/毕设/2023国赛C/数据/大类/4.食用菌/classify_4_dataset_day.csv"),
  data_5 = read.csv("D:/桌面/毕设/2023国赛C/数据/大类/5.花菜/classify_5_dataset_day.csv"),
  data_6 = read.csv("D:/桌面/毕设/2023国赛C/数据/大类/6.茄类/classify_6_dataset_day.csv")
)
pred_feature<- function(data, windows, n_future_days){
  data_selected <- data %>%
    select(total_cost, total_price)
  for (i in 1:n_future_days) {
    last_win_days <- tail(data_selected, windows)
    new_day <- colMeans(last_win_days)
    data_selected <- rbind(data_selected, new_day)
  }
  data_selected <- tail(data_selected, n_future_days)
  return(data_selected)
}
future_predictions <- lapply(data_list, pred_feature, windows = 7, n_future_days = 7)

output_dir <- "D:/桌面/毕设/2023国赛C/数据/大类/预测数据集"  
for (i in seq_along(future_predictions)) {
  file_name <- paste0(output_dir, "pred_", i, ".csv")
  write.csv(future_predictions[[i]], file = file_name, row.names = FALSE)
}


# 每个蔬菜单品的预测特征值的计算
##预测思路一：利用最后出现该产品前7天所有数据的平均值（这样预测有的产品不应季不会卖）
##预测思路二：采用前7天的数据来预测未来会卖啥产品（这样预测不会预测到新产品）
##预测思路三：采用前一个月的数据的平均值来决定未来卖啥产品
