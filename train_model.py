import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib

# Tải dữ liệu
train_df = pd.read_csv("dataset/train_survival.csv")
test_df = pd.read_csv("dataset/test_survival.csv")

# Tách các đặc trưng và giá trị churn từ các file
X_train = train_df.drop(columns=['churn_value'])
y_train = train_df['churn_value']

X_test = test_df.drop(columns=['churn_value'])
y_test = test_df['churn_value']

# Tải dữ liệu survival
survival_train = pd.read_csv("dataset/survival_features_train.csv")
survival_test = pd.read_csv("dataset/survival_features_test.csv")

# Merge lại vào X_train, X_test bằng index (thứ tự hàng)
X_train_final = pd.concat([X_train.reset_index(drop=True), survival_train], axis=1)
X_test_final = pd.concat([X_test.reset_index(drop=True), survival_test], axis=1)

X_train_final = pd.get_dummies(X_train_final, columns=['hazard_group'], drop_first=False)
X_test_final = pd.get_dummies(X_test_final, columns=['hazard_group'], drop_first=False)

# Sử dụng RobustScaler cho các cột số liệu
columns_to_scale = ['hazard_score', 'baseline_hazard', 'survival_prob_3m', 'survival_prob_6m', 'survival_prob_12m']

# Khởi tạo RobustScaler
scaler = RobustScaler()

# Chuẩn hóa các cột cho cả train và test
X_train_final[columns_to_scale] = scaler.fit_transform(X_train_final[columns_to_scale])
X_test_final[columns_to_scale] = scaler.transform(X_test_final[columns_to_scale])

# Huấn luyện mô hình (Ví dụ: RandomForest)
model = RandomForestClassifier()
model.fit(X_train_final, y_train)

# Lưu mô hình vào file churn_model.pkl
joblib.dump(model, "churn_project/churn_model/churn_model.pkl")
