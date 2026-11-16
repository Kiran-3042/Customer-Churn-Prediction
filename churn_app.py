import streamlit as st
import pandas as pd
import numpy as np
import joblib



st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in=False

if not st.session_state.logged_in:

    with st.container(border=True):
        st.title("Customer Churn Prediction📊")
        st.header("Login Page")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.stop()
st.sidebar.title("Navigation")

option=st.sidebar.selectbox(
    "Choose a page",
    ["Customer Information","Customer Services","Billing Information"]
)



model=joblib.load("churn_model.pkl")
st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave this company")
st.success("Machine Learning model loaded successfully")

st.header("Customer Information")
gender=st.selectbox(
    "Select Your Gender",
    ["Male","Female","Other"]
)

senior_citizen=st.selectbox(
    "Select your senior citizen",
    [0,1]
)

Partner=st.radio(
    "Select Your Partner",
    ["Yes","No"]
)

Dependents=st.radio(
    "Select Your Dependents",
    ["Yes","No"]
)

tenure=st.number_input(
    "Select Your Tenure",
    min_value=0,
    max_value=72,
    value=72
)

st.header("Customer Services")
phone_service=st.selectbox(
    "Select your phone service",
    ["Yes","No"]
)

multiple_lines=st.selectbox(
    "Select your multiple lines",
    ["Yes","No","No phone service"]
)

internet_Service=st.selectbox(
    "Select your internet service",
    ["DSL","Fiber optic","No"]
)

online_security=st.selectbox(
    "Select your online security",
    ["Yes","No","No internet service"]
)

online_backup=st.selectbox(
    "Select your online backup",
    ["Yes","No","No internet service"]
)

device_protection=st.selectbox(
    "Select your device protection",
    ["Yes","No","No internet service"]
)

tech_support=st.selectbox(
    "Select your tech support",
    ["Yes","No","No internet service"]
)

streaming_tv=st.selectbox(
    "Select your streaming tv",
    ["Yes","No","No internet service"]
)

streaming_movies=st.selectbox(
    "Select your streaming movies",
    ["Yes","No","No internet service"]
)

st.header("Billing Information")
contract=st.selectbox(
    "Select your contract",
    ["Month-to-month","One year","Two year"]
)

paperless_billing=st.selectbox(
    "Select your Paperless billing",
    ["Yes","No"]
)

payment_method = st.selectbox(
    "Select your Payment method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges=st.number_input("Enter a Monthly charges",
                                min_value=0.0,step=0.01)

total_charges=st.number_input("Enter a Total charges",min_value=0.0,step=0.01)

df=pd.DataFrame([{
    "gender":gender,
    "SeniorCitizen":senior_citizen,
    "Partner":Partner,
    "Dependents":Dependents,
    "tenure":tenure,
    "PhoneService":phone_service,
    "MultipleLines":multiple_lines,
    "InternetService":internet_Service,
    "OnlineSecurity":online_security,
    "OnlineBackup":online_backup,
    "DeviceProtection":device_protection,
    "TechSupport":tech_support,
    "StreamingTV":streaming_tv,
    "StreamingMovies":streaming_movies,
    "Contract":contract,
    "PaperlessBilling":paperless_billing,
    "PaymentMethod":payment_method,
    "MonthlyCharges":monthly_charges,
    "TotalCharges":total_charges

}])


prediction=model.predict(df)

probabilities=model.predict_proba(df)



model_classes=model.classes_


if prediction[0]=="Yes":
    st.warning("The Customer is likely to leave")
else:
    st.success("The Customer is not likely to leave")

churn_probabilities=probabilities[0][1]

st.metric("Churn Probability",f"{churn_probabilities*100:.2f}%")

col1,col2,col3=st.columns(3)

with col1:
    st.metric("Churn Prediction:",prediction[0])

with col2:
    st.metric("Churn Probability:",f"{churn_probabilities*100:.2f}%")

with col3:
    st.metric("Customer Tenure:",tenure,delta="Months")

task=pd.DataFrame({
    "Class":model.classes_,
    "Probabilities":probabilities[0]
})

st.bar_chart(task,x="Class",y="Probabilities")

analytics_data=pd.read_csv("Customer.csv")

avg_charges=analytics_data.groupby("Churn")["MonthlyCharges"].mean().reset_index()

st.subheader("Average Monthly Charges by Churn Status")
st.bar_chart(avg_charges,x="Churn",y="MonthlyCharges")