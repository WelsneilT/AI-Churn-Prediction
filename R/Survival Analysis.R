# Library
library(survival)
library(survminer)
library(dplyr)

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

# Tính predicted survival

df_survival <- df_survival %>%
  rowwise() %>%
  mutate(
    base_hazard_6m = get_cumulative_hazard(tenure + 6),
    base_hazard_12m = get_cumulative_hazard(tenure + 12),
    base_hazard_24m = get_cumulative_hazard(tenure + 24),
    predicted_survival_tenure_6m = exp(-base_hazard_6m * exp(hazard_score)),
    predicted_survival_tenure_12m = exp(-base_hazard_12m * exp(hazard_score)),
    predicted_survival_tenure_24m = exp(-base_hazard_24m * exp(hazard_score))
  ) %>%
  ungroup()

# Làm tròn kết quả để dễ quan sát
df_survival <- df_survival %>%
  mutate(across(starts_with("predicted_survival_tenure"), ~ round(.,4))) %>%
  mutate(across(starts_with("predicted_survival_tenure"), ~ ifelse(. < 0.0001, 0, .)))

# Đánh giá mô hình bằng Harrell's C-index
concordance(cox_model)

# Chia nhóm hazard và vẽ Kaplan-Meier

# Chia thành 4 nhóm theo hazard score
df_survival$hazard_group <- cut(df_survival$hazard_score, breaks=quantile(df_survival$hazard_score), include.lowest=TRUE)

# Vẽ survival curves theo hazard group
fit <- survfit(Surv(tenure, churn_value) ~ hazard_group, data = df_survival)
ggsurvplot(fit = fit, data = df_survival,
           risk.table = TRUE,
           ggtheme = theme_minimal(),
           title = "Survival Curves by Hazard Group")

# Kiểm tra sự khác biệt giữa các nhóm bằng Log-rank test
surv_diff <- survdiff(surv_obj ~ hazard_group, data = df_survival)
print(surv_diff)

# Visualize yếu tố ảnh hưởng tới hazard
ggforest(cox_model, data = df_survival)




