# churn_app/views.py

import os
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from .forms import UploadFileForm, OnlinePredictionForm
import warnings

# Suppress specific warnings if needed, e.g., FutureWarning from older sklearn/pandas interactions
# warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
MODEL_DIR = os.path.join(settings.BASE_DIR, 'churn_model')
# Only core models needed now, scalers will be created dynamically
CORE_MODEL_FILES = ['catboost_model.pkl', 'cox_model.pkl', 'kmeans_model.pkl']

# --- Global Variables ---
catboost_model = None
cox_model = None
kmeans_model = None
baseline_hazard_df = None # Loaded from cox_model
cox_feature_names_from_model = None
kmeans_feature_names_from_model = None
# No global scaler variables needed anymore
models_loaded = False
model_load_error = None

# Define column groups (Based on notebook analysis)
COLS_TO_DROP_INITIAL = [
    'churn_score', 'churn_label', 'churn_category', 'churn_reason',
    'customer_status', 'satisfaction_score', 'cltv', 'married',
    'referred_a_friend', 'total_revenue', 'churn_value',
    'country', 'state', # Assuming these were dropped or not used by final models
]
# Columns to scale *before* survival features (based on notebook scaling `df`)
# Note: In the notebook, these were scaled on the *training* data (`df`).
# Replicating the notebook *exactly* for the test set is tricky as it merged pre-scaled data.
# We will apply scaling *within* preprocess_data, but fitting on the input batch.
NORMAL_COLS_SCALING_PRE = ['monthly_charges', 'avg_monthly_long_distance_charges', 'total_charges']
OUTLIER_COLS_SCALING_PRE = [
    'number_of_referrals', 'number_of_dependents', 'avg_monthly_gb_download',
    'total_refunds', 'total_extra_data_charges', 'total_long_distance_charges',
    'total_population', 'age'
]
# Columns to scale *after* survival features (based on notebook scaling `user_df_merged`)
SURVIVAL_COLS_TO_SCALE = ['hazard_score', 'baseline_hazard', 'survival_prob_3m', 'survival_prob_6m', 'survival_prob_12m']

# --- Load Core Models ---
try:
    print(f"INFO: Attempting to load core models from: {MODEL_DIR}")
    missing_files = [f for f in CORE_MODEL_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing required model files: {', '.join(missing_files)}")

    catboost_model = joblib.load(os.path.join(MODEL_DIR, 'catboost_model.pkl'))
    cox_model = joblib.load(os.path.join(MODEL_DIR, 'cox_model.pkl'))
    kmeans_model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))

    if hasattr(cox_model, 'baseline_hazard_'):
        baseline_hazard_df = cox_model.baseline_hazard_
        baseline_hazard_df.index = pd.to_numeric(baseline_hazard_df.index, errors='coerce')
        baseline_hazard_df = baseline_hazard_df.dropna().sort_index()
        print(f"INFO: Baseline hazard loaded. Index type: {baseline_hazard_df.index.dtype}, Shape: {baseline_hazard_df.shape}")
    else:
        baseline_hazard_df = None
        print("WARNING: Cox model missing baseline_hazard_ attribute.")

    try: cox_feature_names_from_model = list(cox_model.feature_names_in_)
    except AttributeError: print("WARNING: Cox model missing 'feature_names_in_'.")
    try: kmeans_feature_names_from_model = list(kmeans_model.feature_names_in_)
    except AttributeError: print("WARNING: KMeans model missing 'feature_names_in_'.")

    models_loaded = True
    print("INFO: Core models loaded successfully.")

except FileNotFoundError as e: model_load_error = f"Model Loading Error: {e}."
except Exception as e: model_load_error = f"Unexpected model loading error: {e}."
if model_load_error: print(f"ERROR: {model_load_error}"); models_loaded = False

# --- Helper Functions ---
def get_cumulative_hazard(tenure, baseline_hazard_df):
    """Calculates cumulative baseline hazard up to a given tenure."""
    if baseline_hazard_df is None or tenure is None or pd.isna(tenure): return 0.0
    try:
        tenure = pd.to_numeric(tenure, errors='coerce')
        if pd.isna(tenure): return 0.0
        valid_baseline = baseline_hazard_df.loc[baseline_hazard_df.index <= tenure]
        return float(valid_baseline['baseline hazard'].sum()) if not valid_baseline.empty else 0.0
    except Exception as e: print(f"ERROR: Cumulative hazard calc failed for tenure {tenure}: {e}"); return 0.0

def get_survival_probabilities(row, baseline_hazard_df):
    """Calculates survival probabilities using Cox formula."""
    default_probs = (0.0, 0.0, 0.0)
    if baseline_hazard_df is None or 'tenure' not in row or 'hazard_score' not in row: return default_probs
    try:
        current_tenure = pd.to_numeric(row['tenure'], errors='coerce')
        hazard_score_lp = pd.to_numeric(row['hazard_score'], errors='coerce') # Linear predictor (beta*X)
        if pd.isna(current_tenure) or pd.isna(hazard_score_lp): return default_probs

        # Hazard Ratio = exp(Linear Predictor)
        # Use np.exp cautiously, handle potential overflow
        try: hazard_ratio = np.exp(np.clip(hazard_score_lp, -700, 700)) # Clip to avoid large exponents
        except OverflowError: hazard_ratio = np.inf

        if np.isinf(hazard_ratio): # If hazard ratio is infinite, survival is 0
             print(f"WARNING: Infinite hazard ratio for hazard_score {hazard_score_lp}. Survival prob set to 0.")
             return 0.0, 0.0, 0.0

        # Cumulative Baseline Hazard at future points H0(t)
        cum_hazard_3m = get_cumulative_hazard(current_tenure + 3, baseline_hazard_df)
        cum_hazard_6m = get_cumulative_hazard(current_tenure + 6, baseline_hazard_df)
        cum_hazard_12m = get_cumulative_hazard(current_tenure + 12, baseline_hazard_df)

        # Survival Probability S(t|X) = exp(-H0(t) * Hazard_Ratio)
        prob_3m = np.exp(np.clip(-cum_hazard_3m * hazard_ratio, -700, 0)) # Clip negative exponent
        prob_6m = np.exp(np.clip(-cum_hazard_6m * hazard_ratio, -700, 0))
        prob_12m = np.exp(np.clip(-cum_hazard_12m * hazard_ratio, -700, 0))

        return np.clip(prob_3m, 0, 1), np.clip(prob_6m, 0, 1), np.clip(prob_12m, 0, 1)

    except Exception as e: print(f"ERROR: Survival probability calc failed: {e}"); return default_probs

# --- Preprocessing Function ---
def preprocess_data(df_raw):
    """
    Applies the full preprocessing pipeline, dynamically fitting scalers
    and applying KMeans to mimic the Colab notebook's final steps on the test set.
    """
    if not models_loaded: raise RuntimeError(model_load_error or "Models not loaded.")

    df = df_raw.copy()
    print(f"DEBUG: Preprocessing started. Input shape: {df.shape}")
    customer_ids = df['customer_id'].copy() if 'customer_id' in df.columns else None
    if customer_ids is None: df['customer_id'] = [f"TEMP_ID_{i}" for i in range(len(df))]

    # === Step 1-6: Drops, Imputation, Encoding, Feature Engineering, OHE ===
    # (This part should be identical to the notebook's processing of the *initial* data)
    print("DEBUG: Applying Steps 1-6 (Drops, Impute, Encode, Feature Eng, OHE)...")
    cols_to_drop_now = [col for col in COLS_TO_DROP_INITIAL if col in df.columns]
    df = df.drop(columns=cols_to_drop_now, errors='ignore')
    if 'offer' in df.columns: df['offer'] = df['offer'].fillna('No Offer')
    if 'internet_type' in df.columns: df['internet_type'] = df['internet_type'].fillna('Unknown')
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        if 'tenure' in df.columns: df['total_charges'] = df.apply(lambda r: 0 if pd.isna(r['total_charges']) and r['tenure'] == 0 else r['total_charges'], axis=1)
        df['total_charges'] = df['total_charges'].fillna(0)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any(): df[col] = df[col].fillna(0)
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}; binary_cols_encoded = []
    for col in df.select_dtypes(include=['object']).columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({'Yes', 'No'}) or set(unique_vals).issubset({'Male', 'Female'}):
            df[col] = df[col].map(binary_map).fillna(0).astype(int); binary_cols_encoded.append(col)
    # Feature Engineering
    if 'tenure' in df.columns:
        bins = [-np.inf, 12, 24, 36, 48, 60, np.inf]; labels = ['1', '2', '3', '4', '5', '6']
        df["New_tenure_year"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False).astype(str).fillna('Unknown')
    if 'contract' in df.columns: df["New_contract_type"] = df["contract"].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2}).fillna(0).astype(int)
    if 'partner' in df.columns and 'dependents' in df.columns:
        df['partner'] = pd.to_numeric(df['partner'], errors='coerce').fillna(0).astype(int)
        df['dependents'] = pd.to_numeric(df['dependents'], errors='coerce').fillna(0).astype(int)
        df["New_family_size"] = df["partner"] + df["dependents"] + 1
    df['New_total_services'] = 0; service_cols_map = {'online_security': 1, 'online_backup': 1, 'device_protection': 1, 'premium_tech_support': 1, 'streaming_tv': 1, 'streaming_movies': 1, 'streaming_music': 1, 'internet_service': 1, 'phone_service': 1}
    for col, value in service_cols_map.items():
        if col in df.columns:
            if col == 'internet_service': df['New_total_services'] += df[col].apply(lambda x: value if x != 'No' else 0)
            else: df['New_total_services'] += df[col].map({'Yes': 1, 1: 1}).fillna(0) * value
    if 'payment_method' in df.columns: auto_payment = ["Bank transfer (automatic)", "Credit card (automatic)"]; df["New_flag_auto_payment"] = df["payment_method"].apply(lambda x: 1 if x in auto_payment else 0).astype(int)
    if 'monthly_charges' in df.columns and 'New_total_services' in df.columns:
        df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce').fillna(0)
        df["New_avg_service_fee"] = df.apply(lambda x: x["monthly_charges"] / x["New_total_services"] if x["New_total_services"] > 0 else x["monthly_charges"], axis=1).replace([np.inf, -np.inf], 0).fillna(0)
    protection_cols = ['online_security', 'online_backup', 'device_protection', 'premium_tech_support']
    if all(c in df.columns for c in protection_cols):
        df['protection_count'] = df[protection_cols].apply(lambda row: sum(row.map({'Yes': 1, 1: 1}).fillna(0)), axis=1)
        df["New_no_protection"] = df['protection_count'].apply(lambda x: 1 if x < len(protection_cols) else 0).astype(int)
        df = df.drop(columns=['protection_count'])
    # Label Encoding
    cols_to_label_encode = []
    for col in df.columns:
         if col not in binary_cols_encoded and col != 'customer_id' and col != 'churn_value':
             col_dtype = df[col].dtype
             if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype):
                 if df[col].nunique(dropna=True) == 2: cols_to_label_encode.append(col)
    if cols_to_label_encode: le = LabelEncoder();
    for col in cols_to_label_encode: df[col] = le.fit_transform(df[col].astype(str))
    # OHE
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist(); cols_to_ohe = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'New_tenure_year' in df.columns and 'New_tenure_year' not in cols_to_ohe: cols_to_ohe.append('New_tenure_year')
    cols_to_ohe = [c for c in cols_to_ohe if c != 'customer_id' and c not in numeric_cols]
    if cols_to_ohe: df = pd.get_dummies(df, columns=cols_to_ohe, drop_first=True, dummy_na=False)
    print(f"DEBUG: Steps 1-6 completed. Shape after OHE: {df.shape}")

    # === Step 7: Scaling Initial Numerics (Fit on current data) ===
    # Mimics notebook scaling the main df, but applied to the input batch/row
    print("DEBUG: Applying Step 7 (Scaling Initial Numerics - Fit on Input)...")
    current_num_cols = df.select_dtypes(include=np.number).columns.tolist()
    current_num_cols = [c for c in current_num_cols if c not in ['customer_id', 'churn_value']]
    engineered_numeric_to_scale = ['New_total_services', 'New_avg_service_fee', 'New_family_size', 'number_of_dependents', 'New_contract_type', 'New_flag_auto_payment', 'New_no_protection']
    current_outlier_cols = [c for c in OUTLIER_COLS_SCALING_PRE if c in current_num_cols]
    for eng_col in engineered_numeric_to_scale:
        if eng_col in current_num_cols and eng_col not in current_outlier_cols: current_outlier_cols.append(eng_col)
    current_normal_cols = [c for c in NORMAL_COLS_SCALING_PRE if c in current_num_cols]

    if current_normal_cols:
        try: scaler_norm = StandardScaler(); df[current_normal_cols] = scaler_norm.fit_transform(df[current_normal_cols])
        except Exception as e: print(f"ERROR: StandardScaler (pre-survival) failed: {e}")
    if current_outlier_cols:
        try: scaler_out = RobustScaler(); df[current_outlier_cols] = scaler_out.fit_transform(df[current_outlier_cols])
        except Exception as e: print(f"ERROR: RobustScaler (pre-survival) failed: {e}")
    print(f"DEBUG: Step 7 completed.")

    # === Step 8: Survival Feature Calculation ===
    print("DEBUG: Applying Step 8 (Survival Feature Calculation)...")
    # Fallback list - **VERIFY THIS AGAINST YOUR COX MODEL TRAINING DATA**
    cox_input_cols_needed_fallback = [
         'contract_One year', 'contract_Two year', 'number_of_referrals', 'number_of_dependents',
         'monthly_charges', 'New_avg_service_fee', 'dependents', 'age', 'latitude', 'city',
         'internet_type_Fiber optic', 'internet_type_Unknown', 'New_family_size_2', 'New_family_size_3',
         'total_charges', 'total_population', 'payment_method_Credit card (automatic)',
         'payment_method_Electronic check', 'payment_method_Mailed check', 'longitude', 'zip_code',
         'New_contract_type_1', 'New_contract_type_2', 'avg_monthly_gb_download', 'senior_citizen'
    ]
    cox_input_cols_to_use = cox_feature_names_from_model if cox_feature_names_from_model else cox_input_cols_needed_fallback
    missing_for_cox = [c for c in cox_input_cols_to_use if c not in df.columns]
    if missing_for_cox: print(f"WARNING: Adding missing Cox columns: {missing_for_cox}");
    for c in missing_for_cox: df[c] = 0
    if cox_model and baseline_hazard_df is not None and 'tenure' in df.columns:
        try:
            df_cox_input = df[cox_input_cols_to_use].apply(pd.to_numeric, errors='coerce').fillna(0)
            df['hazard_score'] = cox_model.predict_partial_hazard(df_cox_input)
            df['baseline_hazard'] = df['tenure'].apply(lambda t: get_cumulative_hazard(t, baseline_hazard_df))
            probs = df.apply(lambda row: get_survival_probabilities(row, baseline_hazard_df), axis=1)
            df['survival_prob_3m'], df['survival_prob_6m'], df['survival_prob_12m'] = zip(*probs)
        except Exception as e: print(f"ERROR: Survival calculation failed: {e}");
        # Ensure columns exist even if calculation failed
        for col in SURVIVAL_COLS_TO_SCALE: df[col] = df.get(col, 0.0)
    else: print(f"WARNING: Skipping survival calculation.");
    for col in SURVIVAL_COLS_TO_SCALE: df[col] = df.get(col, 0.0) # Ensure columns exist
    print(f"DEBUG: Step 8 completed.")
    # print(df[SURVIVAL_COLS_TO_SCALE].head()) # Optional debug

    # === Step 9: Create and OHE Hazard Group ===
    print("DEBUG: Applying Step 9 (Hazard Group Creation/OHE)...")
    hazard_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']; hazard_group_cols_expected = [f'hazard_group_{lbl}' for lbl in hazard_labels]
    try:
        if 'hazard_score' in df.columns and pd.api.types.is_numeric_dtype(df['hazard_score']) and df['hazard_score'].notna().all():
             # Calculate quantiles *on the current input data* to mimic notebook
             q_values = df['hazard_score'].quantile([0.25, 0.5, 0.75])
             bins = sorted(list(set([-np.inf, q_values[0.25], q_values[0.50], q_values[0.75], np.inf])))
             if len(bins) < 3: df['hazard_group'] = 'Low' # Handle low variance case
             else:
                 current_labels = hazard_labels[:len(bins)-1]
                 df['hazard_group'] = pd.cut(df['hazard_score'], bins=bins, labels=current_labels, right=False, duplicates='drop')
                 df['hazard_group'] = df['hazard_group'].cat.add_categories('Unknown').fillna('Unknown')
             df = pd.get_dummies(df, columns=['hazard_group'], prefix='hazard_group', drop_first=False)
        else: print("WARNING: 'hazard_score' missing/non-numeric/NaN. Skipping Hazard Group.");
        for col in hazard_group_cols_expected: df[col] = df.get(col, 0) # Ensure columns exist
    except Exception as e: print(f"ERROR: Hazard Group creation failed: {e}");
    for col in hazard_group_cols_expected: df[col] = df.get(col, 0) # Ensure columns exist
    print(f"DEBUG: Step 9 completed.")

    # === Step 10: Scaling Survival Features (Fit on current data) ===
    # Mimics notebook scaling user_df_merged's survival columns
    print("DEBUG: Applying Step 10 (Scaling Survival Features - Fit on Input)...")
    present_survival_cols = [col for col in SURVIVAL_COLS_TO_SCALE if col in df.columns]
    if present_survival_cols:
        try:
            df[present_survival_cols] = df[present_survival_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            # Instantiate and fit RobustScaler *on the current data*
            scaler_surv_dynamic = RobustScaler()
            df[present_survival_cols] = scaler_surv_dynamic.fit_transform(df[present_survival_cols])
            print(f"DEBUG: Applied dynamic RobustScaler to survival features: {present_survival_cols}")
        except Exception as e: print(f"ERROR: Dynamic RobustScaler for survival features failed: {e}")
    else: print("DEBUG: No survival features present to scale.")
    print(f"DEBUG: Step 10 completed.")
    # print(df[present_survival_cols].head()) # Optional debug

    # === Step 11: KMeans Clustering (Predict only) ===
    # Using predict, as fit_predict on test data is unusual.
    print("DEBUG: Applying Step 11 (KMeans Prediction)...")
    kmeans_cluster_col_name = 'kmeans_cluster'
    if kmeans_model:
        try:
            if kmeans_feature_names_from_model: kmeans_input_features = kmeans_feature_names_from_model
            else: exclude_for_kmeans = ['customer_id', 'churn_value']; kmeans_input_features = df.select_dtypes(include=np.number).columns.tolist(); kmeans_input_features = [col for col in kmeans_input_features if col not in exclude_for_kmeans]; print(f"WARNING: Inferring KMeans features ({len(kmeans_input_features)}).")
            missing_kmeans = [c for c in kmeans_input_features if c not in df.columns];
            if missing_kmeans: print(f"WARNING: Adding missing KMeans columns: {missing_kmeans}");
            for c in missing_kmeans: df[c] = 0
            kmeans_input = df[kmeans_input_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            # Use predict, not fit_predict
            df[kmeans_cluster_col_name] = kmeans_model.predict(kmeans_input)
            print(f"DEBUG: KMeans prediction applied.")
        except Exception as e: print(f"ERROR: KMeans prediction failed: {e}"); df[kmeans_cluster_col_name] = 0
    else: print("WARNING: KMeans model not loaded."); df[kmeans_cluster_col_name] = 0
    print(f"DEBUG: Step 11 completed.")
    # print(df[[kmeans_cluster_col_name]].head()) # Optional debug

    # === Step 12: Final Feature Reconstruction & Alignment ===
    print("DEBUG: Applying Step 12 (Final Feature Alignment)...")
    FINAL_EXPECTED_FEATURES = None
    if hasattr(catboost_model, 'feature_names_') and catboost_model.feature_names_:
         FINAL_EXPECTED_FEATURES = catboost_model.feature_names_
         print(f"INFO: Using feature list from loaded CatBoost model ({len(FINAL_EXPECTED_FEATURES)} features).")
    else:
         # --- FALLBACK: HARDCODED LIST - **VERIFY AND COMPLETE THIS** ---
         print(f"WARNING: CatBoost model missing feature names. Using hardcoded list. VERIFY THIS IS CORRECT!")
         FINAL_EXPECTED_FEATURES = [
             # Add ALL columns exactly as they appear in the notebook's `user_df_merged`
             # right before the `catboost_model.predict` call.
             # This needs meticulous checking against your notebook.
             'gender', 'age', 'under_30', 'senior_citizen', 'partner', 'dependents', 'number_of_dependents', 'tenure',
             'internet_service', 'phone_service', 'multiple_lines', 'avg_monthly_gb_download', 'unlimited_data',
             'number_of_referrals', 'online_security', 'online_backup', 'device_protection', 'premium_tech_support',
             'streaming_tv', 'streaming_movies', 'streaming_music', 'paperless_billing', 'monthly_charges',
             'avg_monthly_long_distance_charges', 'total_charges', 'total_refunds', 'total_extra_data_charges',
             'total_long_distance_charges', 'city', 'zip_code', 'total_population', 'latitude', 'longitude',
             'New_total_services', 'New_flag_auto_payment', 'New_avg_service_fee', 'New_no_protection',
             'offer_Offer B', 'offer_Offer C', 'offer_Offer D', 'offer_Offer E', # Example OHE
             'internet_type_Fiber optic', 'internet_type_Unknown', # Example OHE
             'contract_One year', 'contract_Two year', # Example OHE
             'payment_method_Credit card (automatic)', 'payment_method_Electronic check', 'payment_method_Mailed check', # Example OHE
             'New_tenure_year_2', 'New_tenure_year_3', 'New_tenure_year_4', 'New_tenure_year_5', 'New_tenure_year_6', # Example OHE
             'New_contract_type_1', 'New_contract_type_2', # Example OHE
             'New_family_size_2', 'New_family_size_3', # Example OHE
             'hazard_score', 'baseline_hazard', 'survival_prob_3m', 'survival_prob_6m', 'survival_prob_12m',
             'hazard_group_Low', 'hazard_group_Medium-Low', 'hazard_group_Medium-High', 'hazard_group_High',
             'kmeans_cluster'
         ]
         # Clean up the list (remove duplicates, sort for consistency if needed)
         FINAL_EXPECTED_FEATURES = sorted(list(set(FINAL_EXPECTED_FEATURES)))

    # Final Type Conversions
    for col in df.columns:
         if col in FINAL_EXPECTED_FEATURES or col == kmeans_cluster_col_name:
             col_dtype = df[col].dtype
             if pd.api.types.is_bool_dtype(col_dtype): df[col] = df[col].astype(int)
             elif pd.api.types.is_object_dtype(col_dtype):
                 unique_vals = set(df[col].dropna().unique())
                 if unique_vals.issubset({True, False, 'True', 'False', '1', '0', 1, 0}): df[col] = df[col].map({True: 1, 'True': 1, 1: 1, False: 0, 'False': 0, 0: 0}).fillna(0).astype(int)
                 else: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
             elif pd.api.types.is_float_dtype(col_dtype):
                  try: # Check if float can be safely converted to int
                      if np.array_equal(df[col].dropna(), df[col].dropna().astype(int)): df[col] = df[col].fillna(0).astype(int)
                  except: pass # Keep as float if conversion fails

    # Align columns
    current_cols = set(df.columns); expected_cols_set = set(FINAL_EXPECTED_FEATURES)
    missing_final = list(expected_cols_set - current_cols); extra_final = list(current_cols - expected_cols_set - {'customer_id'})
    if missing_final: print(f"WARNING: Adding missing final columns: {missing_final}");
    for c in missing_final: df[c] = 0
    if extra_final: print(f"DEBUG: Removing extra final columns: {extra_final}"); df = df.drop(columns=extra_final, errors='ignore')
    try:
        df_processed = df[FINAL_EXPECTED_FEATURES] # Reorder and select ONLY expected features
        print("DEBUG: Final column alignment successful.")
    except KeyError as e: raise ValueError(f"Final column alignment failed. Missing columns: {e}")

    print(f"DEBUG: Preprocessing finished. Final shape for prediction: {df_processed.shape}")
    return df_processed, customer_ids

# === Django Views (predict_online, predict_batch - Keep as in previous answer) ===
# No changes needed in the view functions themselves, they call preprocess_data

def predict_online(request):
    """Handles single prediction requests from the online form."""
    page_title = 'Online Churn Prediction'; prediction_mode = 'online'
    context = {'prediction_mode': prediction_mode, 'page_title': page_title}
    if not models_loaded: context['error'] = model_load_error or 'Core models unavailable.'; context['form'] = OnlinePredictionForm(); return render(request, 'churn_app/prediction_online.html', context)
    if request.method == 'POST':
        form = OnlinePredictionForm(request.POST)
        if form.is_valid():
            try:
                df_raw = pd.DataFrame([form.cleaned_data])
                print("\n--- Processing Online Prediction ---")
                X_predict, _ = preprocess_data(df_raw) # Preprocess

                # Ensure columns are in the correct order before prediction (Safety check)
                if hasattr(catboost_model, 'feature_names_') and catboost_model.feature_names_:
                    try: X_predict = X_predict[catboost_model.feature_names_]
                    except KeyError as e: raise ValueError(f"Column mismatch after preprocessing for online prediction: {e}")

                prediction = catboost_model.predict(X_predict)[0]
                probability = catboost_model.predict_proba(X_predict)[0, 1]
                context['prediction_value'] = int(prediction)
                context['prediction_probability'] = float(probability)
                result_text = "Prediction: CHURN" if prediction == 1 else "Prediction: NO CHURN"
                context['prediction_result'] = f"{result_text} (Probability: {probability:.3f})"
                print(f"INFO: Online Prediction Result: {context['prediction_result']}")
            except (RuntimeError, ValueError, KeyError, TypeError) as e: context['error'] = f"Prediction Error: {e}"; print(f"ERROR: {context['error']}")
            except Exception as e: context['error'] = f"Unexpected error: {e}"; print(f"ERROR: {context['error']}")
        context['form'] = form
    else: context['form'] = OnlinePredictionForm()
    return render(request, 'churn_app/prediction_online.html', context)

def predict_batch(request):
    """Handles batch prediction requests from CSV file uploads."""
    page_title = 'Batch Churn Prediction'; prediction_mode = 'batch'
    context = {'prediction_mode': prediction_mode, 'page_title': page_title}
    if not models_loaded: context['error'] = model_load_error or 'Core models unavailable.'; context['form'] = UploadFileForm(); return render(request, 'churn_app/prediction_batch.html', context)
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['csv_file']; original_filename = file.name
            context['original_filename'] = original_filename
            try:
                if not original_filename.lower().endswith('.csv'): raise ValueError("Invalid file type.")
                try: df_raw = pd.read_csv(file, encoding='utf-8-sig')
                except UnicodeDecodeError: file.seek(0); df_raw = pd.read_csv(file, encoding='latin1')
                except Exception as read_e: raise ValueError(f"Could not read CSV: {read_e}")
                if 'customer_id' not in df_raw.columns: raise ValueError("'customer_id' column missing.")
                print(f"\n--- Processing Batch Prediction for {original_filename} ({len(df_raw)} rows) ---")
                X_predict, customer_ids = preprocess_data(df_raw) # Preprocess

                # Ensure columns are in the correct order before prediction (Safety check)
                if hasattr(catboost_model, 'feature_names_') and catboost_model.feature_names_:
                     try: X_predict = X_predict[catboost_model.feature_names_]
                     except KeyError as e: raise ValueError(f"Column mismatch after preprocessing for batch prediction: {e}")

                predictions = catboost_model.predict(X_predict)
                probabilities = catboost_model.predict_proba(X_predict)[:, 1]
                results_df = pd.DataFrame({'Customer ID': customer_ids, 'Prediction (1=Churn)': predictions.astype(int), 'Churn Probability': probabilities})
                results_df['Churn Probability'] = results_df['Churn Probability'].map('{:.3f}'.format)
                context['results_html'] = results_df.to_html(classes=['table', 'table-striped', 'table-hover', 'table-sm', 'table-bordered'], index=False, justify='center', border=0)
                print(f"INFO: Batch prediction completed for {len(results_df)} customers.")
            except (RuntimeError, ValueError, KeyError, TypeError) as e: context['error'] = f"Prediction Error: {e}"; print(f"ERROR: {context['error']}")
            except Exception as e: context['error'] = f"Error processing file '{original_filename}': {e}"; print(f"ERROR: {context['error']}")
        context['form'] = form
    else: context['form'] = UploadFileForm()
    return render(request, 'churn_app/prediction_batch.html', context)