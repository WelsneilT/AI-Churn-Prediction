library(ggplot2)
library(dplyr)
library(knitr)
library(kableExtra)
# Đọc dữ liệu
df <- read.csv("D:/Documents/IAI-UET/AI - Churn Prediction/dataset/Merged_Churn_Data.csv")  # Thay bằng đường dẫn thực tế

# Tạo nhóm tuổi dựa trên "Tenure in Months"
df <- df %>%
  mutate(Tenure_Group = case_when(
    Tenure.in.Months < 12 ~ "New Customer",
    Tenure.in.Months >= 12 & Tenure.in.Months < 24 ~ "1-2 Years",
    Tenure.in.Months >= 24 & Tenure.in.Months < 48 ~ "2-4 Years",
    Tenure.in.Months >= 48 ~ "4+ Years"
  ))

# Tạo bảng tổng hợp số lượng churn theo nhóm tuổi
df_summary <- df %>%
  group_by(Tenure_Group, churn_label) %>%
  summarise(Count = n()) %>%
  pivot_wider(names_from = churn_label, values_from = Count, values_fill = 0) %>%
  rename("Không rời mạng" = `No`, "Rời mạng" = `Yes`) %>%
  mutate("Tổng số khách hàng" = `Không rời mạng` + `Rời mạng`,
         "Tỷ lệ churn (%)" = round(`Rời mạng` / `Tổng số khách hàng` * 100, 2))

# Hiển thị bảng dưới dạng đẹp
kable(df_summary, caption = "Bảng tỷ lệ churn theo nhóm tuổi") %>%
  kable_styling(latex_options = c("striped", "hold_position"))
