import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from agent import investigation_report
from model_training import trained_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ------------------------------------------------------ user_input------------------------------------------
dataset_version = Path("synthetic_fraud_dataset.csv").stat().st_mtime_ns
ohe_cols, xg, encoder, sc, x, xts_sc, y_test = trained_data(dataset_version)

def UI():
    st.markdown("<h2 style='color:yellow; text-align:center;'>Credit Card Fraud Detection System</h2>", unsafe_allow_html=True)
    st.divider()

    with st.form("credit_card form"):
        st.markdown("<h4 style='color:lightgreen'>Enter the details below</h4>", unsafe_allow_html=True)
        trs_amount = st.number_input("Enter the Trasaction amount: ", min_value=0)
        trs_type = st.selectbox("Enter the type of the Transaction: ", ['POS', 'Bank Transfer', 'Online', 'ATM Withdrawal'])
        acc_balance = st.number_input("Enter the Account Balance: ", min_value=0, max_value=1000000)
        device = st.selectbox('Enter the Device used for the payment: ',[None,'Laptop', 'Mobile', 'Tablet'])
        card_loc = st.selectbox("Enter the location: ", ['London', 'Mumbai', 'New York', 'Sydney', 'Tokyo'])
        hour = st.number_input('Enter the payment hour: ', min_value=0)
        date = st.number_input('Enter the payment date: ', min_value=0)
        month = st.number_input('Enter the payment month: ', min_value=0)
        isnight = st.selectbox('is the payment done at night? ', ['Yes', 'No'])
        isnight = 1 if isnight == 'Yes' else 0
        day_of_week = st.number_input('Enter the payment day of week: ', min_value=0)
        merchant_cat = st.selectbox("Enter the expense field: ",  ['Travel' ,'Clothing' , 'Restaurants' , 'Electronics' ,'Groceries'])
        auth_method = st.selectbox("Enter the authentication method: ",  ['Biometric' ,'Password', 'OTP' ,'PIN'])
        card_type = st.selectbox("Enter the card type: ",  ['Amex' ,'Mastercard', 'Visa','Discover'])
        prev_act = st.selectbox("Enter the previous fraudalent activity: ", ["Yes", "No"])
        prev_act = 1 if prev_act == 'Yes' else 0
        ip_flagged = st.selectbox("is the IP Address Flagged: ", ['Yes', 'No'])
        ip_flagged = 1 if ip_flagged == 'Yes' else 0
        daily_tr_count = st.number_input("Enter the daily transaction count: ", min_value=0)
        avg_amount = st.number_input("Enter the Average Transaction amount in 7 days: ", min_value=0)
        failed_tr_count = st.number_input("Enter the count of failed transaction in 7 days: ", min_value=0)
        is_weekend = st.selectbox("is the transaction did on weekend or not : ", ['Yes', 'No'])
        is_weekend = 1 if is_weekend == 'Yes' else 0
        card_age = st.number_input("Enter the Card Age: ", min_value=0)
        risk_score = st.number_input("Enter the Risk Score: ")
        tr_distance = st.number_input("Enter the transaction distance: ")
        submit = st.form_submit_button('submit', type='primary', use_container_width=True)


    user_data = {
        'Transaction_Amount' : trs_amount,
        'Transaction_Type' : trs_type,
        'Account_Balance' : acc_balance,
        'Device_Type' : device,
        'Location' : card_loc,
        'Hour' : hour,
        'Date' : date,
        'Month' : month,
        'is_night' : isnight,
        'Dayofweek' : day_of_week,
        "Merchant_Category" : merchant_cat,
        'Authentication_Method' : auth_method,
        'Card_Type' : card_type,
        'Previous_Fraudulent_Activity' : prev_act,
        'IP_Address_Flag': ip_flagged,
        'Daily_Transaction_Count' : daily_tr_count,
        'Avg_Transaction_Amount_7d' : avg_amount,
        'Failed_Transaction_Count_7d' : failed_tr_count,
        'Is_Weekend' : is_weekend,
        'Card_Age' : card_age,
        'Risk_Score' : risk_score,
        'Transaction_Distance' : tr_distance
    }

    input_data = pd.DataFrame([user_data])


    user_encode = encoder.transform(input_data[ohe_cols])
    user_encoded_df = pd.DataFrame(user_encode, columns = encoder.get_feature_names_out(ohe_cols))
    user_df = pd.concat([input_data.drop(columns= ohe_cols, errors='ignore'), user_encoded_df], axis=1)
    user_df = user_df.reindex(columns=x.columns, fill_value=0)
    user_scaled = sc.transform(user_df)

    if submit:
        user_pred = xg.predict(user_scaled)[0]
        pred_lable = "Fraud" if user_pred == 1 else "Not Fraud"
        if user_pred == 0:
            print(user_pred)
            st.success("The Card is not fraud ✔")
        else:
            st.error("The Card is Fraud ⚠")

    # ------------------------------------------------------model predictions------------------------------------------
        model_pred = xg.predict(xts_sc)

    # ------------------------------------------------------interpretatins------------------------------------------
        st.markdown("<h4 style='color:red; text-align:center;'>XGBOOST</h4>", unsafe_allow_html=True)
        acc = st.write("Accuracy: ", round(accuracy_score(y_test, model_pred)*100, 2),"%")
        prec_score = st.write("Precision Score: ", round(precision_score(y_test, model_pred)*100,2))
        rec_score = st.write("Recall Score: ", round(recall_score(y_test, model_pred)*100, 2))
        f1 = st.write ("f1 score: ", round(f1_score(y_test, model_pred)*100, 2))
        roc_auc = st.write("ROC-AUC Score: ", round(roc_auc_score(y_test, model_pred)*100, 2))
        con_matrix = st.write("Confusion Matrix: ", confusion_matrix(y_test, model_pred))

    # -------------------------------------------------AI Agent Investigation Report------------------------------------------------------------
        with st.sidebar:
            st.markdown("<h3 style='color:orange; text-align:center;'>AI Agent Investigation Report</h3>", unsafe_allow_html=True)
            with st.spinner("Generating Investigation Report..."):
                try:
                    report = investigation_report(user_input=user_data, prediction_label=pred_lable)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"The generation is stopped. Check the backend for error: {e}")

if __name__ == "__main__":
    UI()