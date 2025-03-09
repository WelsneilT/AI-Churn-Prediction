library(survival)   # Chạy mô hình Cox
library(survminer)  # Vẽ biểu đồ survival analysis
library(dplyr)      # Xử lý dữ liệu
library(ggplot2)    # Đồ họa

# Đọc dữ liệu
df <- read.csv("D:/Downloads/Merged_Churn_Data1.csv")

# Chọn các biến quan trọng
df_survival <- df %>%
  select(tenure, churn_value, monthly_charge, total_charges, contract, satisfaction_score, cltv, internet_type, age, number_of_referrals) %>%
  mutate(
    churn_value = as.numeric(churn_value),  # Chuyển churn_value thành số (0 hoặc 1)
    contract = as.factor(contract),         # Chuyển contract thành categorical
    internet_type = as.factor(internet_type) # Chuyển internet_type thành categorical
  )

# Xóa các dòng có NA để tránh lỗi
df_survival <- na.omit(df_survival)

# Kiểm tra dữ liệu sau xử lý
str(df_survival)


# Tạo Survival Object
surv_obj <- Surv(time = df_survival$tenure, event = df_survival$churn_value)

# Chạy Cox Model với nhiều feature
cox_model <- coxph(surv_obj ~ monthly_charge + total_charges + contract + satisfaction_score + cltv + internet_type + age + number_of_referrals, 
                   data = df_survival)

# Hiển thị kết quả mô hình
summary(cox_model)
# Kaplan-Meier Curve theo loại hợp đồng
km_fit <- survfit(surv_obj ~ contract, data = df_survival)

# Vẽ biểu đồ
ggsurvplot(
  km_fit, 
  data = df_survival,
  pval = TRUE, conf.int = TRUE,  # Hiển thị p-value và khoảng tin cậy
  risk.table = TRUE,             # Hiển thị bảng số lượng khách hàng còn lại
  risk.table.title = "Số khách hàng còn lại theo thời gian",
  risk.table.height = 0.25,       # Điều chỉnh kích thước bảng
  risk.table.col = "strata",      # Màu bảng giống màu biểu đồ
  legend.title = "Loại hợp đồng",
  palette = c("#E15759", "#76B7B2", "#4E79A7"),  # Màu sắc tùy chỉnh
  ggtheme = theme_minimal(base_size = 14),  # Theme nhẹ nhàng, chữ lớn hơn
  title = "Phân tích sống sót của khách hàng theo loại hợp đồng",
  xlab = "Thời gian (tháng)", 
  ylab = "Xác suất sống sót",
  surv.median.line = "hv"  # Vẽ đường chỉ median survival
)
