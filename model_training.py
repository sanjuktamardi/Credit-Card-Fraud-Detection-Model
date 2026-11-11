# ------------------------------------------------------Libraries------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import streamlit as st

# ------------------------------------------------------Loading the data------------------------------------------
@st.cache_resource
def trained_data(dataset_version):
    data = pd.read_csv("synthetic_fraud_dataset.csv")

    print("Data loaded succesfully!!\n", data.head())
    print("Data details:\n", data.describe)
    print("Data Type:\n", data.dtypes)
    print("columns: ", data.columns)
    print('='*150)

    # ------------------------------------------------------Data Preprocessing------------------------------------------
    missing_val = data.isnull().sum()
    print("Totals missing values: ",data.isnull().sum().sum())
    print("missing values:\n",missing_val)

    duplicate_val = data.duplicated().sum()
    print("Duplicated missing values: ", duplicate_val)
    print('='*150)

    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['Date'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month
    data['Dayofweek'] = data['Timestamp'].dt.day_of_week
    data['is_night'] = data['Hour'].apply(lambda x: 1 if x<6 or x>22 else 0)

    # ------------------------------------------------------Feature Engineering------------------------------------------
    ohe_cols = ['Merchant_Category', 'Authentication_Method', 'Transaction_Type', 'Device_Type', 'Location', 'Card_Type' ]
    omit_columns = ['Transaction_ID', 'User_ID', 'Timestamp']

    # ------------------------------------------------------one-hot encoding------------------------------------------
    # one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(data[ohe_cols])
    encoded_cols = encoder.get_feature_names_out(ohe_cols)
    encoded_df = pd.DataFrame(encoded_data, columns = encoded_cols)
    df = pd.concat([data.drop(columns = ohe_cols + omit_columns), encoded_df], axis=1)
    print(df.head())
    print(df.dtypes)
    # df.to_csv("cleaned_df.csv")

    # ------------------------------------------------------selecting feature and targets------------------------------------------
    x = df.drop(columns = ['Fraud_Label'])
    y = df['Fraud_Label']

    # ------------------------------------------------------train-test-split------------------------------------------
    x = df.drop(columns=['Fraud_Label', 'Transaction_Distance', 'Risk_Score'])
    y = df['Fraud_Label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # ------------------------------------------------------Model train------------------------------------------
    sc = StandardScaler()
    x_sc = sc.fit_transform(x_train)
    xts_sc = sc.transform(x_test)

    xg = XGBClassifier(
        n_estimators=500,
        learning_rate=0.10,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model = xg.fit(x_sc, y_train)

    return ohe_cols, xg, encoder, sc, x, xts_sc, y_test