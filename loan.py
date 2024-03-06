import streamlit as st
import pandas as pd
import numpy as np
from joblib import load 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE 

#Load the model
model=load('Loan_Default_App.joblib')
 #Function to preprocess input data
def preprocess_data(age, income, loan_amount, credit_score, months_employed, credit_lines, interest_rate,
                    loan_term, dti_ratio, education, employment_type, marital_status, has_mortgage, loan_purpose,
                    has_cosigner, has_dependents):
    age = [age]
    income = [income]
    loan_amount = [loan_amount]
    credit_score = [credit_score]
    months_employed = [months_employed]
    credit_lines = [credit_lines]
    interest_rate = [interest_rate]
    loan_term = [loan_term]
    dti_ratio = [dti_ratio]
    education = [education]
    employment_type = [employment_type]
    marital_status = [marital_status]
    has_mortgage = [has_mortgage]
    loan_purpose = [loan_purpose]
    has_cosigner = [has_cosigner]
    has_dependents = [has_dependents]

    loan_default = pd.DataFrame({
        'Age': age, 'Income': income, 'LoanAmount': loan_amount,
        'CreditScore': credit_score, 'MonthsEmployed': months_employed,
        'NumCreditLines': credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education,
        'EmploymentType': employment_type, 'MaritalStatus': marital_status,
        'HasMortgage': has_mortgage, 'LoanPurpose': loan_purpose,
        'HasCosigner': has_cosigner, 'Has_Dependents': has_dependents})
    
    # Map to encode the categorical columns
    loan_default['HasCosigner'] = loan_default['HasCosigner'].map({'No': 0, 'Yes': 1})
    loan_default['HasMortgage'] = loan_default['HasMortgage'].map({'No': 0, 'Yes': 1})
    loan_default['Has_Dependents'] = loan_default['Has_Dependents'].map({'No': 0, 'Yes': 1})
    loan_default['Education'] = loan_default['Education'].map({"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3})
    loan_default['EmploymentType'] = loan_default['EmploymentType'].map({"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3})
    loan_default['MaritalStatus'] = loan_default['MaritalStatus'].map({"Divorced": 0, "Married": 1, "Single": 2})
    loan_default['LoanPurpose'] = loan_default['LoanPurpose'].map({"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4})
    
    # Scale the input variables
    scaler = StandardScaler()
    columns = ['InterestRate', 'DTIRatio', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines']

    loan_default[columns] = scaler.fit_transform(loan_default[columns])
    
    return loan_default

## streamlit layout
def main():
    st.title('Loan Default Prediction Application')
    st.image('loan_image.jpg', use_column_width=True)
    st.write('This is a Loan Prediction Application that tells you if a customer will default loan payment')
    st.subheader('Enter the details below to predict loan default')
  
    # Get user input
    age = st.number_input('Age', value=30, step=1)
    income = st.number_input('Income', value=1000000, step=1000)
    loan_amount = st.number_input('Loan Amount', value=10000, step=100)
    credit_score = st.number_input('Credit Score', value=300, step=10)
    months_employed = st.number_input('Months Employed', value=12, step=1)
    credit_lines = st.number_input('Number of Credit Lines', value=2, step=1)
    interest_rate = st.number_input('Interest Rate (%)', value=10, step=1)
    loan_term = st.slider('Loan Term (months)', min_value=12, max_value=360, value=120, step=12)
    dti_ratio = st.slider('DTI Ratio', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    education = st.selectbox("Education", ["Bachelor's", "Master's", 'High School', 'PhD'])
    employment_type = st.selectbox('Employment Type', ['Full-time', 'Unemployed', 'Self-employed', 'Part-time'])
    marital_status = st.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
    has_mortgage = st.radio("Has Mortgage", ['No', 'Yes'])
    loan_purpose = st.selectbox('Loan Purpose', ['Other', 'Auto', 'Business', 'Home', 'Education'])
    has_cosigner = st.radio("Has Cosigner", ['No', 'Yes'])
    has_dependents = st.radio("Has Dependents", ['No', 'Yes'])

    # Preprocess the user input
    user_data=preprocess_data(age, income, loan_amount, credit_score, months_employed, credit_lines, interest_rate,
                    loan_term, dti_ratio, education, employment_type, marital_status, has_mortgage,
                    loan_purpose, has_cosigner, has_dependents)
    
# Make predictions with the loaded model
    prediction = model.predict(user_data)
    # Display the Predictions
    st.write('Model Prediction', prediction)
    st.write('1: Means client will default loan')
    st.write('0: Means client will pay back loan')
    result = 'Yes,Loan will default' if prediction[0] == 1 else 'No,Loan will be paid back'
    st.write(prediction[0], result)


if __name__ == "__main__":
    main()



