# Customer Churn Prediction for Subscription-Based Service

### 📌 Overview
This project aims to predict **customer churn** for a subscription-based streaming service. Customer churn is a critical metric for businesses offering subscription services, as it impacts revenue and long-term customer engagement. By leveraging **predictive analytics**, businesses can anticipate churn risks and implement strategies to retain customers.

### 📂 Dataset Description
The dataset consists of anonymized customer subscription details and their interaction with the streaming service. It includes information on subscription plans, payment methods, viewing behavior, customer support interactions, and other relevant attributes.

### 📁 Files in the Dataset
- `train.csv` - Training dataset with customer details and churn labels.
- `test.csv` - Testing dataset without churn labels for model evaluation.
- `data_descriptions.csv` - Description of dataset columns and their meanings.

### 📊 Columns in the Dataset
| Column Name | Type | Data Type | Description |
|------------|------|----------|-------------|
| `CustomerID` | Identifier | string | A unique identifier for each customer. |
| `AccountAge` | Feature | integer | The age of the user's account in months. |
| `MonthlyCharges` | Feature | float | The amount charged to the user on a monthly basis. |
| `TotalCharges` | Feature | float | The total charges incurred by the user over the account's lifetime. |
| `SubscriptionType` | Feature | object | The type of subscription chosen by the user (Basic, Standard, or Premium). |
| `PaymentMethod` | Feature | string | The method of payment used by the user. |
| `PaperlessBilling` | Feature | string | Indicates whether the user has opted for paperless billing (Yes or No). |
| `ContentType` | Feature | string | The type of content preferred by the user (Movies, TV Shows, or Both). |
| `MultiDeviceAccess` | Feature | string | Indicates whether the user has access to the service on multiple devices (Yes or No). |
| `DeviceRegistered` | Feature | string | The type of device registered by the user (TV, Mobile, Tablet, or Computer). |
| `ViewingHoursPerWeek` | Feature | float | The number of hours the user spends watching content per week. |
| `AverageViewingDuration` | Feature | float | The average duration of each viewing session in minutes. |
| `ContentDownloadsPerMonth` | Feature | integer | The number of content downloads by the user per month. |
| `GenrePreference` | Feature | string | The preferred genre of content chosen by the user. |
| `UserRating` | Feature | float | The user's rating for the service on a scale of 1 to 5. |
| `SupportTicketsPerMonth` | Feature | integer | The number of support tickets raised by the user per month. |
| `Gender` | Feature | string | The gender of the user (Male or Female). |
| `WatchlistSize` | Feature | float | The number of items in the user's watchlist. |
| `ParentalControl` | Feature | string | Indicates whether parental control is enabled for the user (Yes or No). |
| `SubtitlesEnabled` | Feature | string | Indicates whether subtitles are enabled for the user (Yes or No). |
| `Churn` | Target | integer | The target variable indicating whether a user has churned or not (1 for churned, 0 for not churned). |

---
### 🎯 Objective
The main objective of this project is to predict **customer churn** based on their subscription details, payment history, and engagement with the service. Identifying customers with a high risk of churn enables businesses to:

✅ Implement **personalized retention strategies**
✅ Optimize **customer engagement and content recommendations**
✅ Improve **pricing and subscription plans**
✅ Enhance **customer support services**

---
### 🚀 Use Cases
- **Churn Prediction Models** - Identify high-risk customers likely to cancel subscriptions.
- **Retention Strategies** - Develop targeted campaigns to retain at-risk customers.
- **Personalized Recommendations** - Suggest relevant content to increase engagement.
- **Pricing Strategy Optimization** - Adjust subscription plans based on customer behavior.

---
### 🛠 Next Steps
1. **Data Preprocessing** - Handle missing values, encode categorical variables, and normalize numerical features.
2. **Feature Engineering** - Create new meaningful features to improve model performance.
3. **Model Training & Evaluation** - Implement **Machine Learning models** (Logistic Regression, Decision Tree, XGBoost) and **Deep Learning models** (LSTM) for better predictions.
4. **Deployment & Monitoring** - Deploy the model and continuously monitor its performance.

This project aims to leverage data-driven insights to **enhance customer retention efforts** and **optimize subscription services**. 📈🎬


