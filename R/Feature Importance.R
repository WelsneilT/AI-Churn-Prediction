library(survival)
library(dplyr)

# Đọc dữ liệu và chuẩn hóa
df <- read.csv("D:/Downloads/Merged_Churn_Data.csv")

# Chọn 15 biến quan trọng nhất từ mô hình Random Forest
df_survival <- df %>%
  select(customer_id, tenure, churn_value, contract, monthly_charges, total_revenue, total_charges,
         internet_service, number_of_referrals, total_long_distance_charges, age,
         latitude, avg_monthly_gb_download, longitude, cltv, total_population, internet_type) %>%
  mutate(
    churn_value = as.numeric(churn_value),
    contract = as.factor(contract),
    internet_service = as.factor(internet_service),
    internet_type = as.factor(internet_type)
  ) %>%
  na.omit()

# Tạo Survival object
surv_obj <- Surv(time = df_survival$tenure, event = df_survival$churn_value)

# Xây dựng mô hình Cox với top 15 feature
cox_model <- coxph(surv_obj ~ contract + monthly_charges + total_revenue + total_charges +
                     internet_service + number_of_referrals + total_long_distance_charges + age +
                     latitude + avg_monthly_gb_download + longitude + cltv + total_population + internet_type,
                   data = df_survival)

# Kết quả mô hình Cox
summary(cox_model)

# Tính baseline hazard
base_surv <- basehaz(cox_model, centered = FALSE)

# Hàm lấy cumulative hazard tại tenure khách hàng
get_cumulative_hazard <- function(time_point){
  idx <- max(which(base_surv$time <= time_point))
  return(base_surv$hazard[idx])
}

# Tính các đặc trưng survival cần xuất
df_export <- df_survival %>%
  mutate(
    hazard_score = predict(cox_model, newdata = df_survival, type = "lp"),
    baseline_hazard = sapply(tenure, get_cumulative_hazard),
    hazard_group = cut(hazard_score, 
                       breaks=quantile(hazard_score, probs = seq(0, 1, 0.25)),
                       labels=c("Low", "Medium-Low", "Medium-High", "High"), 
                       include.lowest=TRUE)
  ) %>%
  select(customer_id, hazard_score, baseline_hazard, hazard_group)

# Xuất ra file CSV
write.csv(df_export, "survival_features.csv", row.names = FALSE)
