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
km_fit <- survfit(surv_obj ~ internet_type, data = df_survival)

ggsurvplot(
  km_fit, 
  data = df_survival,
  pval = TRUE, conf.int = TRUE,  # Hiển thị p-value và khoảng tin cậy
  risk.table = TRUE,             # Hiển thị bảng số lượng khách hàng còn lại
  risk.table.title = "Số khách hàng còn lại theo thời gian",
  risk.table.height = 0.3,        # Điều chỉnh chiều cao bảng
  risk.table.col = "strata",      # Màu bảng giống màu biểu đồ
  risk.table.y.text = TRUE,       # Hiển thị nhãn của từng nhóm internet_type
  risk.table.fontsize = 4,      # Tăng kích thước chữ trong bảng
  legend.title = "Loại Internet",
  legend.labs = c("Cable", "DSL", "Fiber Optic", "None"),  # Định dạng lại nhãn legend
  palette = c("#E15759", "#76B7B2", "#4E79A7", "#F28E2B"),  # Màu sắc tương phản
  title = "Phân tích sống sót của khách hàng theo loại Internet",
  surv.median.line = "hv",
  ggtheme = theme_minimal(base_size = 14) + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))  # Căn giữa title
)

