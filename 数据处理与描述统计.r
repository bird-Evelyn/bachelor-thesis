library(tidyverse)
library(readxl)
library(proxy)
library(e1071)
library(ggthemes)  
library(scales)    
single_info <- read_excel("附件1.xlsx")
data <- read_excel("附件2.xlsx")
single_cost <- read_excel("附件3.xlsx")

## 1) 整合数据
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
   }
  
  lapply(1:6, function(x) save_classify(data, x))
  
  
  data <- read.csv("data.csv")
  
  # 每月所有蔬菜的总销售量
  month_total_qty <- data %>% 
    mutate(year_month = as.Date(paste(year, month, '01', sep = "-")))%>% 
    select(-c(year,month)) %>%
    group_by(year_month) %>% 
    summarise(month_qty = sum(qty), 
              .groups = 'drop') 
  ggplot(month_total_qty, aes(x = year_month, y = month_qty)) +
    geom_area(
      fill = "#AED6F1", 
      alpha = 0.4
    ) +  
    geom_line(
      color = "#1B4F72", 
      linewidth = 2
    ) + 
    geom_point(
      color = "#1B4F72", 
      fill = "#F1948A", 
      size = 3.5, 
      shape = 21, 
      stroke = 1.2
    ) +  
    scale_x_date(
      date_labels = "%Y-%m",
      date_breaks = "2 months",  
      expand = expansion(mult = c(0.02, 0.02))  
    ) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.05)) 
    ) +
    labs(
      title = "每月蔬菜总需求量变化趋势",
      subtitle = "数据周期：2020年7月 - 2023年6月",
      x = "时间（年月）",
      y = "需求量总和（单位：斤）"
    ) +
    theme_minimal(base_size = 16) +  
    theme(
      plot.title = element_text(
        hjust = 0.5, face = "bold", size = 22, color = "#2C3E50"
      ),  
      plot.subtitle = element_text(
        hjust = 0.5, size = 16, color = "gray40"
      ),
      plot.caption = element_text(
        hjust = 1, size = 10, color = "gray50"
      ),
      axis.title.x = element_text(
        margin = margin(t = 10), face = "bold"
      ),
      axis.title.y = element_text(
        margin = margin(r = 10), face = "bold"
      ),
      axis.text.x = element_text(
        angle = 45, hjust = 1, vjust = 1, size = 12, color = "#34495E"
      ),
      axis.text.y = element_text(
        size = 12, color = "#34495E"
      ),
      panel.grid.major.x = element_blank(), 
      panel.grid.minor = element_blank(), 
      panel.grid.major.y = element_line(
        color = "gray85", linetype = "dashed"
      ),
      plot.background = element_rect(
        fill = "white", color = NA
      ),  
      panel.background = element_rect(
        fill = "white", color = NA
      )
    )
  
  # 六大总类的月总销量随时间变换的图像
  month_qty <- data %>%
    mutate(year_month = as.Date(paste(year, month, '01', sep = "-")))%>% 
    select(-c(year,month)) %>% 
    group_by(classify_name, year_month) %>% 
    summarise(month_qty = sum(qty), 
              .groups = 'drop') 
  ggplot(month_qty, aes(x = year_month, y = month_qty, color = classify_name)) +
    geom_line(linewidth = 1.5) +  # 折线加粗，1.5更清晰
    geom_point(size = 3, shape = 21, stroke = 1.2, fill = "white") +  
    scale_color_brewer(palette = "Set1") +  
    scale_x_date(
      limits = as.Date(c("2020-07-01", "2023-07-01")),
      date_labels = "%Y-%m",
      date_breaks = "3 months",
      expand = expansion(mult = c(0.02, 0.02)) 
    ) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.05)) 
    ) +
    labs(
      title = "六大品类的月总需求量变化趋势",
      subtitle = "数据周期：2020年7月 - 2023年6月",
      x = "时间（年月）",
      y = "月需求量（单位：斤）",
      color = "产品大类"
    ) +
    theme_minimal(base_size = 16) +
    theme(
      plot.title = element_text(
        hjust = 0.5, face = "bold", size = 22, color = "#2C3E50"
      ),
      plot.subtitle = element_text(
        hjust = 0.5, size = 16, color = "gray40"
      ),
      plot.caption = element_text(
        hjust = 1, size = 10, color = "gray50"
      ),
      axis.title.x = element_text(
        margin = margin(t = 10), face = "bold"
      ),
      axis.title.y = element_text(
        margin = margin(r = 10), face = "bold"
      ),
      axis.text.x = element_text(
        angle = 45, hjust = 1, vjust = 1, size = 12, color = "#34495E"
      ),
      axis.text.y = element_text(
        size = 12, color = "#34495E"
      ),
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 12),
      panel.grid.major.x = element_blank(),  
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "gray85", linetype = "dashed"),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )
  
  # 计算六大类每日需求量的数字特征
  statistics <- data %>% 
    group_by(all_date, classify_name) %>% 
    summarise(total_qty= sum(qty), .groups = "drop") %>% 
    group_by(classify_name) %>% 
    summarise(
      total = sum(total_qty),
      max_qty = max(total_qty),
      min_qty = min(total_qty),
      mean_qty = mean(total_qty),
      median_qty = median(total_qty),
      sd_qty = sd(total_qty),
      skewness_qty = skewness(total_qty),
      kurtosis_qty = kurtosis(total_qty))

  my_colors <- c("#3498DB", "#1ABC9C", "#9B59B6", "#F1C40F", "#E67E22", "#E74C3C")
  
  ggplot(statistics, aes(x = reorder(classify_name, total), y = total, fill = classify_name)) +
    geom_col(width = 0.6, alpha = 0.85) +  
    geom_text(
      aes(label = comma(total)),
      vjust = -0.6, size = 4, color = "black", fontface = "bold"
    ) +
    scale_fill_manual(values = my_colors) +  
    labs(
      title = "六大品类三年总需求量直方图",
      subtitle = "数据周期：2020年7月 - 2023年6月",
      x = "产品类别",
      y = "总需求量（斤）",
      fill = "品类名称"
    ) +
    theme_minimal(base_size = 16) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 24, color = "#2C3E50"),
      plot.subtitle = element_text(hjust = 0.5, size = 16, color = "gray40", margin = margin(b = 15)),
      axis.title.x = element_text(face = "bold", size = 14, margin = margin(t = 10)),
      axis.title.y = element_text(face = "bold", size = 14, margin = margin(r = 10)),
      axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1, size = 12, color = "#34495E"),
      axis.text.y = element_text(size = 12, color = "#34495E"),
      legend.position = "none",  
      panel.grid.major.y = element_line(color = "gray90", linetype = "dashed"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    ) +
    scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.1))) 
  
  
  