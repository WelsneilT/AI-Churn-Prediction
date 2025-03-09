library(randomForest)
library(dplyr)
library(forcats)
library(caret)
library(ggplot2)
library(tidyr)

# 🔹 Bước 1: Đọc dữ liệu
df <- read.csv("D:/Downloads/Merged_Churn_Data.csv")

# 🔹 Bước 2: Loại bỏ các cột không cần thiết (trừ churn_value!)
df <- df %>%
  select(-customer_id, -city, -state, -zip_code, -churn_reason, 
         -churn_category, -churn_score, -customer_status, 
         -churn_label, -satisfaction_score)

# 🔹 Kiểm tra lại cột churn_value có tồn tại không?
colnames(df)  # churn_value phải xuất hiện ở đây

# 🔹 Bước 3: Xử lý giá trị thiếu (NA)
df <- df %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)),
         across(where(is.character), ~ replace_na(.x, "Unknown")))


# 🔹 Kiểm tra lại churn_value lần nữa
colnames(df)  # churn_value vẫn phải ở đây

# 🔹 Bước 5: Chuyển churn_value sang factor
df$churn_value <- as.factor(df$churn_value)

# 🔹 Kiểm tra churn_value hợp lệ
table(df$churn_value)

# 🔹 Bước 6: Chạy mô hình Random Forest
set.seed(123)
rf_model <- randomForest(churn_value ~ ., data = df, importance = TRUE, ntree = 100)

# 🔹 Bước 7: Lấy bảng biến quan trọng
importance_table <- data.frame(importance(rf_model))
importance_table$Feature <- rownames(importance_table)

# 🔹 Bước 8: Sắp xếp độ quan trọng giảm dần
importance_table <- importance_table %>%
  arrange(desc(MeanDecreaseGini))

# 🔹 Bước 9: Trực quan hóa
ggplot(importance_table, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)",
       x = "Feature",
       y = "Mean Decrease Gini") +
  theme_minimal()

# 🔹 Bước 10: In bảng importance đầy đủ
print(importance_table)
