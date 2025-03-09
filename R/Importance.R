# Library
# Library đầy đủ cần thiết
library(survival)
library(survminer)
library(dplyr)
library(ggplot2)

# Đọc và chuẩn hóa dữ liệu
df <- read.csv("D:/Downloads/Merged_Churn_Data.csv")

df_survival <- df %>%
  select(tenure, churn_value, monthly_charges, total_charges, contract, cltv,  
         internet_type, age, number_of_referrals, total_revenue, total_long_distance_charges) %>%
  mutate(
    churn_value = as.numeric(churn_value),
    contract = as.factor(contract),
    internet_type = as.factor(internet_type)
  ) %>%
  na.omit()

# Tạo Survival Object
surv_obj <- Surv(time = df_survival$tenure, event = df_survival$churn_value)

# Xây dựng mô hình Cox
cox_model <- coxph(surv_obj ~ monthly_charges + total_charges + contract + 
                     cltv + internet_type + age + number_of_referrals + total_revenue + 
                     total_long_distance_charges, data = df_survival)

summary(cox_model)

# Tính baseline hazard
base_surv <- basehaz(cox_model, centered = FALSE)

# Hàm lấy cumulative hazard tại 1 thời điểm
get_cumulative_hazard <- function(time_point){
  idx <- max(which(base_surv$time <= time_point))
  return(base_surv$hazard[idx])
}

# Tính hazard score (linear predictor)
df_survival$hazard_score <- predict(cox_model, newdata = df_survival, type = "lp")

# Boxplot kiểm tra hazard_score và churn_value
ggplot(df_survival, aes(x = factor(churn_value), y = hazard_score, fill = factor(churn_value))) +
  geom_boxplot() +
  labs(x = "Churn (0 = No, 1 = Yes)", 
       y = "Hazard Score",
       title = "Boxplot of Hazard Score by Churn") +
  theme_minimal()

# Kiểm tra bằng thống kê (Wilcoxon test)
wilcox.test(hazard_score ~ churn_value, data = df_survival)
