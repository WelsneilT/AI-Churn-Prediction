
library(survival)   # Chạy mô hình Cox
library(survminer)  # Vẽ biểu đồ survival analysis
library(dplyr)      # Xử lý dữ liệu

# Đọc dữ liệu
df <- read.csv("D:/Downloads/Merged_Churn_Data1.csv")

# Chọn các biến quan trọng
df_survival <- df %>%
  select(tenure, churn_value, monthly_charge, total_charges, contract, satisfaction_score, cltv, 
         internet_type, age, number_of_referrals) %>%
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
cox_model <- coxph(surv_obj ~ monthly_charge + total_charges + contract + satisfaction_score + cltv + 
                     internet_type + age + number_of_referrals, data = df_survival)

# Hiển thị kết quả mô hình
summary(cox_model)


# Tạo hazard score (Nguy cơ churn)
df_survival$hazard_score <- predict(cox_model, newdata = df_survival, type = "risk")

# Lấy xác suất sống sót tại các mốc thời gian 6, 12, 24 tháng
surv_probs <- summary(survfit(cox_model), times = c(6, 12, 24))$surv

# Gán xác suất sống sót vào dataframe
df_survival$survival_6m <- surv_probs[1]  
df_survival$survival_12m <- surv_probs[2] 
df_survival$survival_24m <- surv_probs[3]

# Kiểm tra DataFrame sau khi thêm feature mới
head(df_survival)

