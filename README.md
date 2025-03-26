# Customer Churn Prediction with Survival Analysis

## 📌 Overview

This project aims to predict **customer churn** for a subscription-based streaming service. Customer churn is a critical metric for businesses offering subscription services, as it impacts revenue and long-term customer engagement. By leveraging **predictive analytics**, businesses can anticipate churn risks and implement strategies to retain customers.

## 📂 Dataset Description

The Telco customer churn dataset contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. It indicates which customers have left, stayed, or signed up for their service. Multiple important demographics are included for each customer, as well as a Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index. It is designed to facilitate analysis of customer behavior and retention strategies.

Customer_Info.csv
### 📁 Customer analysis process
![Churn Analysis Diagram](https://github.com/WelsneilT/Churn-Prediction/blob/master/Churn%20analysis%20process.png)

## 📊 Dataset Columns

| Column Name | Type | Data Type | Description |
|-------------|------|-----------|-------------|
| `customer_id` | Identifier | string | A unique identifier for each customer. |
| `gender` | Feature | string | The gender of the user (Male or Female). |
| `age` | Feature | integer | The age of the user. |
| `under_30` | Feature | boolean | Whether the customer is under 30 years old (Yes/No). |
| `senior_citizen` | Feature | boolean | Whether the customer is a senior citizen (Yes/No). |
| `partner` | Feature | boolean | Whether the customer has a partner (Yes/No). |
| `dependents` | Feature | boolean | Whether the customer has dependents (Yes/No). |
| `number_of_dependents` | Feature | integer | The number of dependents. |
| `married` | Feature | boolean | Whether the customer is married (Yes/No). |
| `tenure` | Feature | integer | The length of time the customer has been with the service (in months). |
| `internet_service` | Feature | string | Type of internet service (Fiber Optic, DSL, etc.). |
| `phone_service` | Feature | boolean | Whether the customer has phone service (Yes/No). |
| `multiple_lines` | Feature | boolean | Whether the customer has multiple lines (Yes/No). |
| `avg_monthly_gb_download` | Feature | float | The average monthly data downloaded in GB. |
| `unlimited_data` | Feature | boolean | Whether the customer has unlimited data (Yes/No). |
| `offer` | Feature | string | The offer the customer has (e.g., discounts). |
| `referred_a_friend` | Feature | boolean | Whether the customer referred a friend (Yes/No). |
| `number_of_referrals` | Feature | integer | The number of referrals made by the customer. |
| `online_security` | Feature | boolean | Whether the customer has online security (Yes/No). |
| `online_backup` | Feature | boolean | Whether the customer has online backup (Yes/No). |
| `device_protection` | Feature | boolean | Whether the customer has device protection (Yes/No). |
| `premium_tech_support` | Feature | boolean | Whether the customer has premium tech support (Yes/No). |
| `streaming_tv` | Feature | boolean | Whether the customer has streaming TV (Yes/No). |
| `streaming_movies` | Feature | boolean | Whether the customer has streaming movies (Yes/No). |
| `streaming_music` | Feature | boolean | Whether the customer has streaming music (Yes/No). |
| `internet_type` | Feature | string | Type of internet connection (e.g., Fiber Optic). |
| `contract` | Feature | string | The type of contract (Month-to-Month, One Year, Two Year). |
| `paperless_billing` | Feature | boolean | Whether the customer opted for paperless billing (Yes/No). |
| `payment_method` | Feature | string | Payment method (e.g., Credit Card, Bank Transfer). |
| `monthly_charges` | Feature | float | The monthly charges billed to the customer. |
| `avg_monthly_long_distance_charges` | Feature | float | The average monthly charges for long-distance calls. |
| `total_charges` | Feature | float | The total charges incurred by the customer over the account's lifetime. |
| `total_refunds` | Feature | float | Total amount refunded to the customer. |
| `total_extra_data_charges` | Feature | float | Total charges for extra data usage. |
| `total_long_distance_charges` | Feature | float | Total charges for long-distance calls. |
| `total_revenue` | Feature | float | The total revenue generated from the customer. |
| `satisfaction_score` | Feature | integer | The satisfaction score provided by the customer. |
| `cltv` | Feature | integer | Customer Lifetime Value (CLTV) for the user. |
| `customer_status` | Feature | string | Current status of the customer (Active, Churned). |
| `churn_score` | Target | integer | The churn score for the customer (0 for not churned, 1 for churned). |
| `churn_label` | Target | string | The label indicating whether the customer has churned (Yes/No). |
| `churn_value` | Feature | integer | The value associated with the churn risk (e.g., high, medium, low). |
| `churn_category` | Feature | string | Category of churn risk (e.g., Critical, At-Risk, etc.). |
| `churn_reason` | Feature | string | The reason for churn if known (e.g., Price, Service Quality, etc.). |
| `country` | Feature | string | The country of the customer. |
| `state` | Feature | string | The state of the customer. |
| `city` | Feature | string | The city of the customer. |
| `zip_code` | Feature | string | The postal code of the customer. |
| `total_population` | Feature | integer | The population of the customer's area. |
| `latitude` | Feature | decimal | Latitude of the customer's location. |
| `longitude` | Feature | decimal | Longitude of the customer's location. |

## 🎯 Objective

The main objective of this project is to predict **customer churn** based on their subscription details, payment history, and engagement with the service. Identifying customers with a high risk of churn enables businesses to:

- Implement **personalized retention strategies**
- Optimize **customer engagement and content recommendations**
- Improve **pricing and subscription plans**
- Enhance **customer support services**

## 🚀 Use Cases

- **Churn Prediction Models**: Identify high-risk customers likely to cancel subscriptions.
- **Retention Strategies**: Develop targeted campaigns to retain at-risk customers.
- **Personalized Recommendations**: Suggest relevant content to increase engagement.
- **Pricing Strategy Optimization**: Adjust subscription plans based on customer behavior.


This project aims to leverage data-driven insights to **enhance customer retention efforts** and **optimize subscription services**. 📈🎬



