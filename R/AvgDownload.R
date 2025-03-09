library(ggplot2)
library(dplyr)
library(knitr)
library(kableExtra)
# Đọc dữ liệu
df <- read.csv("D:/Documents/IAI-UET/AI - Churn Prediction/dataset/Merged_Churn_Data.csv")  # Thay bằng đường dẫn thực tế




# Vẽ biểu đồ boxplot với điểm trung bình (Mean)
ggplot(df, aes(x = churn_label, y = Avg.Monthly.GB.Download)) +
  geom_boxplot(aes(fill = churn_label), alpha = 0.7) +  # Thêm alpha để hộp dễ nhìn hơn
  stat_summary(fun = "mean", geom = "point", color = "red", size = 3) +  # Hiển thị trung bình rõ hơn
  labs(
    title = "Mean Difference in Average Monthly Download",
    x = "Churn",
    y = "Average Monthly Download"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("black", "orange"))  # Sử dụng màu tương phản