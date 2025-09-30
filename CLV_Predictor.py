# Deploy CLV Predictor

# ======================================================
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Streamlit for web app
import streamlit as st
import pickle


# ======================================================

# Main Title
st.write('''
# CUSTOMER LIFETIME VALUE (CLV) PREDICTION
''')

# sidebar
st.sidebar.header("Please input Customers Features")


# ======================================================

# build a function to get user input features
def user_input_feature():

    vehicle_class = st.sidebar.selectbox('Vehicle Class', ('Four-Door Car', 'Two-Door Car', 'SUV', 'Sports Car', 'Luxury SUV', 'Luxury Car'))
    coverage = st.sidebar.selectbox('Coverage', ('Basic', 'Extended', 'Premium'))
    renew_offer_type = st.sidebar.selectbox('Renew Offer Type', ('Offer1', 'Offer2', 'Offer3', 'Offer4'))
    employment_status = st.sidebar.selectbox('Employment Status', ('Employed', 'Unemployed', 'Retired', 'Disabled', 'Medical Leave'))
    marital_status = st.sidebar.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
    education = st.sidebar.selectbox('Education', ('High School or Below', 'College', 'Bachelor', 'Master', 'Doctor'))
    number_of_policies = st.sidebar.slider('Number of Policies', min_value=1, max_value=10, value=2, step=1)
    monthly_premium_auto = st.sidebar.number_input('Monthly Premium Auto', min_value=0, max_value=1000, value=100, step=50)
    total_claim_amount = st.sidebar.number_input('Total Claim Amount', min_value=0, max_value=10000, value=500, step=100)
    income = st.sidebar.number_input('Income', min_value=0, max_value=1000000, value=50000, step=1000)

    # Input for CLV
    df = pd.DataFrame()
    df['Vehicle Class'] = [vehicle_class]
    df['Coverage'] = [coverage]
    df['Renew Offer Type'] = [renew_offer_type]
    df['EmploymentStatus'] = [employment_status]
    df['Marital Status'] = [marital_status]
    df['Education'] = [education]
    df['Number of Policies'] = [number_of_policies]
    df['Monthly Premium Auto'] = [monthly_premium_auto]
    df['Total Claim Amount'] = [total_claim_amount]
    df['Income'] = [income]

    # Ensure correct data types
    df['Number of Policies'] = df['Number of Policies'].astype(object)
    df['Monthly Premium Auto'] = df['Monthly Premium Auto'].astype(int)
    df['Total Claim Amount'] = df['Total Claim Amount'].astype(float)
    df['Income'] = df['Income'].astype(float)

    return df

df_customer = user_input_feature()

# load the model
def load_model(filename):
    return pickle.load(open(filename, 'rb'))

# predict a customer
file_name = 'CLV_prediction_model.sav'
loaded_model = load_model(file_name)

clv = loaded_model.predict(df_customer)

# ======================================================

# make two columns for layout
col1, col2 = st.columns(2)

# left side (col1)
with col1:
    # Tampilkan dataframe hasil user input (customer feature)
    st.subheader("Customer Features:")
    st.write(df_customer.transpose())


# bagian kanan (col2)
with col2:
    threshold = st.number_input("Set the threshold for Customer Lifetime Value", min_value=0, max_value=100000, value=10000, step=500)
    st.write(f"Above {threshold} means HIGH Customer Lifetime Value")
    st.write(f"Below {threshold} means LOW Customer Lifetime Value")

    # Tampilkan hasil prediksi
    st.subheader("Prediction")

    if clv >= threshold:
        st.write(f'This Customer predicted will have HIGH Customer Lifetime Value with value: {clv[0]:.2f}. So, it is worth to accept/keep this customer.')
    else:
        st.write(f'This Customer predicted will have LOW Customer Lifetime Value with value: {clv[0]:.2f}. So, it is NOT worth to accept/keep this customer.')


data_batch = st.file_uploader("Upload a CSV")

if data_batch is not None and data_batch.name.endswith('.csv'):
    st.subheader("Batch Prediction")
    df = pd.read_csv(data_batch)
    
    # Validate columns
    required_columns = df_customer.columns.tolist()
    if not all(col in df.columns for col in required_columns):
        st.error("CSV must include the following columns: " + ", ".join(required_columns))
    else:
        #Select only required columns
        df = df[required_columns]

        # Ensure correct data types
        numeric_cols = ['Monthly Premium Auto', 'Total Claim Amount', 'Income']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        #Fill NaN with median
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

        categorical_cols = [col for col in required_columns if col not in numeric_cols]
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        
        #Fill NaN with mode
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Predict and categorize
        df2 = loaded_model.predict(df)
        df['Predicted Customer Lifetime Value'] = df2
        df['CLV Category'] = df['Predicted Customer Lifetime Value'].apply(lambda x: 'HIGH' if x >= threshold else 'LOW')
        
        # Display top 20 for readability
        st.write(df.head(20))
        
        # Add download button for full result
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Predictions as CSV",
            data=csv,
            file_name="CLV_predictions.csv",
            mime="text/csv"
        )

        # Show Predicted summary statistics
        mean_clv = df['Predicted Customer Lifetime Value'].mean()
        median_clv = df['Predicted Customer Lifetime Value'].median()
        low_clv = (df['CLV Category'] == 'LOW').mean() * 100
        high_clv = (df['CLV Category'] == 'HIGH').mean() * 100
        st.subheader("Prediction Summary Statistics")
        st.write(f"Mean Predicted CLV: ${mean_clv:.2f}")
        st.write(f"Median Predicted CLV: ${median_clv:.2f}")
        st.write(f"With threshold ${threshold}, there are {low_clv:.2f}% that predicted as LOW CLV and {high_clv:.2f}% that predicted as HIGH CLV.")

elif data_batch is not None:
    st.error("Please upload a CSV file.")
