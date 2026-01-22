import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# ----------------------
# 1️⃣ App Title & Description
# ----------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")
st.title("💰 Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval based on applicant details.")

# ----------------------
# 2️⃣ Input Section (Sidebar)
# ----------------------
with st.sidebar:
    st.header("Applicant Information")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100, step=10)
    credit_history = st.radio("Credit History", options=["Yes", "No"])
    employment_status = st.selectbox("Employment Status", options=["Employed", "Self-Employed", "Unemployed"])
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

# ----------------------
# 3️⃣ Main Panel Layout
# ----------------------
st.header("Loan Prediction")
st.write("Select the SVM kernel and check loan eligibility:")

# Create columns for a clean look
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("SVM Model Selection")
    kernel_choice = st.radio("Choose SVM Kernel", options=["Linear SVM", "Polynomial SVM", "RBF SVM"])

with col2:
    st.subheader("Ready to Predict")
    predict_button = st.button("✅ Check Loan Eligibility")

# ----------------------
# Prepare Input for Model
# ----------------------
credit_history_encoded = 1 if credit_history == "Yes" else 0

employment_encoder = LabelEncoder()
employment_encoder.fit(["Employed", "Self-Employed", "Unemployed"])
employment_encoded = employment_encoder.transform([employment_status])[0]

property_encoder = LabelEncoder()
property_encoder.fit(["Urban", "Semiurban", "Rural"])
property_encoded = property_encoder.transform([property_area])[0]

X_input = np.array([[applicant_income, loan_amount, credit_history_encoded, employment_encoded, property_encoded]])

# Map kernel
kernel_map = {"Linear SVM": "linear", "Polynomial SVM": "poly", "RBF SVM": "rbf"}
kernel_used = kernel_map[kernel_choice]

# ----------------------
# 4️⃣ Prediction Logic
# ----------------------
if predict_button:
    # Dummy training data
    X_train = np.array([
        [5000, 100, 1, 0, 0],
        [3000, 50, 0, 1, 1],
        [7000, 200, 1, 0, 2],
        [2500, 100, 0, 2, 1]
    ])
    y_train = np.array([1, 0, 1, 0])

    model = SVC(kernel=kernel_used, probability=True)
    model.fit(X_train, y_train)

    prediction = model.predict(X_input)[0]
    confidence = model.predict_proba(X_input)[0][prediction]

    # ----------------------
    # 5️⃣ Output Section
    # ----------------------
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # Bonus Info in columns
    c1, c2 = st.columns(2)
    c1.info(f"**Confidence Score:** {confidence:.2f}")
    c2.info(f"**Kernel Used:** {kernel_choice}")

    # ----------------------
    # 6️⃣ Business Explanation
    # ----------------------
    explanation = (
        "Based on credit history and income pattern, the applicant is likely to repay the loan." 
        if prediction == 1 
        else "Based on credit history and income pattern, the applicant is unlikely to repay the loan."
    )
    st.markdown(f"**Business Explanation:** {explanation}")
