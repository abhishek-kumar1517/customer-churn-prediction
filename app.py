import streamlit as st
import pandas as pd
import joblib

# loading our trained model and column list
model = joblib.load('models/churn_model.pkl')
model_columns = joblib.load('models/model_columns.pkl')

st.title('Customer Churn Prediction Dashboard')
st.write('Upload raw customer data to find out who is at risk of churning')


def clean_and_encode(df):
    # doing the exact same cleaning we did in the notebook
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # creating the same two features we built earlier
    df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure']
    
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = (df[service_cols] == 'Yes').sum(axis=1)

    # converting yes/no columns to 1/0
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({'No': 0, 'Yes': 1, 0: 0, 1: 1})

    # one hot encoding the multi value columns
    df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService',
                                      'Contract', 'PaymentMethod'], drop_first=True)

    # converting any bool columns to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


uploaded_file = st.file_uploader("Upload customer CSV file", type="csv")

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(raw_data.head())

    cleaned_data = clean_and_encode(raw_data)

    # dropping Churn column if it exists, since that's what we're predicting
    if 'Churn' in cleaned_data.columns:
        cleaned_data = cleaned_data.drop('Churn', axis=1)

    # making sure columns match exactly what model expects
    for col in model_columns:
        if col not in cleaned_data.columns:
            cleaned_data[col] = 0
    cleaned_data = cleaned_data[model_columns]

    # getting churn probability for every customer
    probabilities = model.predict_proba(cleaned_data)[:, 1]
    raw_data_clean = raw_data.loc[cleaned_data.index].copy()
    raw_data_clean['Churn_Risk_%'] = (probabilities * 100).round(2)

    st.write("Predictions:")
    st.dataframe(raw_data_clean.sort_values('Churn_Risk_%', ascending=False))

    high_risk = raw_data_clean[raw_data_clean['Churn_Risk_%'] >= 30]
    revenue_at_risk = high_risk['MonthlyCharges'].sum()

    st.subheader("Business Impact")
    st.write(f"High risk customers: {len(high_risk)}")
    st.write(f"Monthly revenue at risk: ${revenue_at_risk:,.2f}")
    st.write(f"Potential annual savings (35% retention): ${revenue_at_risk * 0.35 * 12:,.2f}")