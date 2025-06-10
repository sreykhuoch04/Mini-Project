# import streamlit as st
# import joblib
# import pandas as pd

# # Load trained XGBoost model
# model = joblib.load('best_svc_model.plk')

# st.set_page_config(page_title="Student Dropout Prediction", layout="centered")
# st.title("🎓 Student Dropout Prediction App")
# st.markdown("Enter the student's details to predict whether they are at risk of dropping out.")

# # --- User Inputs ---
# marital_status = st.number_input("Marital Status (e.g., 1 = Single, 2 = Married)", value=1)
# application_mode = st.number_input("Application Mode", value=1)
# application_order = st.number_input("Application Order", min_value=1, value=1)
# course = st.number_input("Course ID", value=1)
# attendance = st.radio("Attendance Type", [1, 0], format_func=lambda x: "Daytime" if x == 1 else "Evening")
# previous_qualification = st.number_input("Previous Qualification", value=1)
# mother_qualification = st.number_input("Mother's Qualification", value=1)
# father_qualification = st.number_input("Father's Qualification", value=1)
# mother_occupation = st.number_input("Mother's Occupation", value=0)
# father_occupation = st.number_input("Father's Occupation", value=0)
# displaced = st.radio("Displaced", [0, 1])
# debtor = st.radio("Debtor", [0, 1])
# tuition_fees_up_to_date = st.radio("Tuition Fees Up-To-Date", [0, 1])
# gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
# scholarship_holder = st.radio("Scholarship Holder", [0, 1])
# age = st.number_input("Age", min_value=15, max_value=60, value=20)
# avg_enrolled = st.number_input("Average Enrolled ", value=0.0)
# avg_approved = st.number_input("Average Approved ", value=0.0)
# avg_grade = st.number_input("Average Grade", value=0.0)

# # --- Prepare input data ---
# input_data = pd.DataFrame([[
#     marital_status, application_mode, application_order, course,
#     attendance, previous_qualification, mother_qualification, father_qualification,
#     mother_occupation, father_occupation, displaced, debtor,
#     tuition_fees_up_to_date, gender, scholarship_holder, age,
#     avg_enrolled, avg_approved, avg_grade
# ]], columns=[
#     'Marital_status', 'Application_mode', 'Application_order', 'Course',
#     'Daytime/evening_attendance', 'Previous_qualification',
#     'Mother_qualification', 'Father_qualification', 'Mother_occupation',
#     'Father_occupation', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
#     'Gender', 'Scholarship_holder', 'Age', 'avg_enrolled',
#     'avg_approved', 'avg_grade'
# ])

# # --- Predict ---
# if st.button("Predict"):
#     prediction = model.predict(input_data)[0]
#     probabilities = model.predict_proba(input_data)[0]

#     result = "🎓 Graduated" if prediction == 1 else "⚠️ Dropped Out"
#     st.subheader("📊 Prediction Result")
#     st.success(f"The student **{result}**.")

#     st.markdown(f"""
#     **Prediction Confidence:**
#     - Dropout Probability: `{probabilities[0]:.2%}`
#     - Graduation Probability: `{probabilities[1]:.2%}`
#     """)

# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np

# # Set page configuration (must be the first Streamlit command)
# st.set_page_config(page_title="Student Dropout Prediction", layout="centered")

# # Load trained XGBoost model and scaler
# try:
#     model = joblib.load('best_svc_model.plk')
#     scaler = joblib.load('scaler.plk')
#     st.success("Model and scaler loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model or scaler: {e}")

# st.title("🎓 Student Dropout Prediction App")
# st.markdown("Enter the student's details to predict whether they are at risk of dropping out.")

# # --- User Inputs ---
# marital_status = st.number_input("Marital Status (e.g., 1 = Single, 2 = Married)", value=1)
# application_mode = st.number_input("Application Mode", value=1)
# application_order = st.number_input("Application Order", min_value=1, value=1)
# course = st.number_input("Course ID", value=1)
# attendance = st.radio("Attendance Type", [1, 0], format_func=lambda x: "Daytime" if x == 1 else "Evening")
# previous_qualification = st.number_input("Previous Qualification", value=1)
# mother_qualification = st.number_input("Mother's Qualification", value=1)
# father_qualification = st.number_input("Father's Qualification", value=1)
# mother_occupation = st.number_input("Mother's Occupation", value=0)
# father_occupation = st.number_input("Father's Occupation", value=0)
# displaced = st.radio("Displaced", [0, 1])
# debtor = st.radio("Debtor", [0, 1])
# tuition_fees_up_to_date = st.radio("Tuition Fees Up-To-Date", [0, 1])
# gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
# scholarship_holder = st.radio("Scholarship Holder", [0, 1])
# age = st.number_input("Age", min_value=15, max_value=60, value=20)
# avg_enrolled = st.number_input("Average Enrolled ", value=0.0)
# avg_approved = st.number_input("Average Approved ", value=0.0)
# avg_grade = st.number_input("Average Grade", value=0.0)

# # --- Prepare input data ---
# input_data = pd.DataFrame([[
#     marital_status, application_mode, application_order, course,
#     attendance, previous_qualification, mother_qualification, father_qualification,
#     mother_occupation, father_occupation, displaced, debtor,
#     tuition_fees_up_to_date, gender, scholarship_holder, age,
#     avg_enrolled, avg_approved, avg_grade
# ]], columns=[
#     'Marital_status', 'Application_mode', 'Application_order', 'Course',
#     'Daytime/evening_attendance', 'Previous_qualification',
#     'Mother_qualification', 'Father_qualification', 'Mother_occupation',
#     'Father_occupation', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
#     'Gender', 'Scholarship_holder', 'Age', 'avg_enrolled',
#     'avg_approved', 'avg_grade'
# ])

# # --- Scale the entire dataset ---
# if 'scaler' in locals():
#     try:
#         input_data_scaled = scaler.transform(input_data)
#         input_data = pd.DataFrame(input_data_scaled, columns=input_data.columns)
#         st.success("Data scaled successfully!")
#     except ValueError as ve:
#         st.error(f"Scaling error: {ve}. Ensure all 19 features match the training data format and order.")
# else:
#     st.warning("Scaler not loaded; data is not scaled.")

# # --- User-provided true label for validation ---
# true_label = st.selectbox("True Label (for validation)", [0, 1], format_func=lambda x: "Dropout" if x == 0 else "Graduated")

# # --- Predict ---
# if st.button("Predict"):
#     if 'model' in locals():
#         prediction = model.predict(input_data)[0]
#         probabilities = model.predict_proba(input_data)[0]

#         result = "🎓 Graduated" if prediction == 1 else "⚠️ Dropped Out"
#         st.subheader("📊 Prediction Result")
#         st.success(f"The student **{result}**.")

#         st.markdown(f"""
#         **Prediction Confidence:**
#         - Dropout Probability: `{probabilities[0]:.2%}`
#         - Graduation Probability: `{probabilities[1]:.2%}`
#         """)

#         # --- Validation ---
#         is_correct = "✅ Correct" if prediction == true_label else "❌ Incorrect"
#         st.subheader("📝 Validation")
#         st.write(f"Prediction: {result}, True Label: {'Graduated' if true_label == 1 else 'Dropped Out'}, Result: {is_correct}")
#     else:
#         st.error("Model not loaded; please check the files.")

import streamlit as st
import joblib
import pandas as pd

# --- Page configuration ---
st.set_page_config(page_title="Student Dropout Prediction", layout="centered")

# --- Load model and scaler ---
try:
    model = joblib.load("best_random_forest_model.plk")
    scaler = joblib.load("scaler.plk")
    st.success("✅ Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")

# --- Title ---
st.title("🎓 Student Dropout Prediction")
st.markdown("Fill in the student's information to predict dropout risk.")

# --- Input form ---
marital_status = st.number_input("Marital Status (e.g., 1 = Single, 2 = Married)", value=1)
application_mode = st.number_input("Application Mode", value=1)
application_order = st.number_input("Application Order", min_value=1, value=1)
course = st.number_input("Course ID", value=1)
attendance = st.radio("Attendance Type", [1, 0], format_func=lambda x: "Daytime" if x == 1 else "Evening")
previous_qualification = st.number_input("Previous Qualification", value=1)
mother_qualification = st.number_input("Mother's Qualification", value=1)
father_qualification = st.number_input("Father's Qualification", value=1)
mother_occupation = st.number_input("Mother's Occupation", value=0)
father_occupation = st.number_input("Father's Occupation", value=0)
displaced = st.radio("Displaced", [0, 1])
debtor = st.radio("Debtor", [0, 1])
tuition_fees_up_to_date = st.radio("Tuition Fees Up-To-Date", [0, 1])
gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
scholarship_holder = st.radio("Scholarship Holder", [0, 1])
age = st.number_input("Age", min_value=15, max_value=60, value=20)
avg_enrolled = st.number_input("Average Enrolled Credits", value=0.0)
avg_approved = st.number_input("Average Approved Credits", value=0.0)
avg_grade = st.number_input("Average Grade", value=0.0)

# --- Construct input data ---
input_data = pd.DataFrame([[
    marital_status, application_mode, application_order, course,
    attendance, previous_qualification, mother_qualification, father_qualification,
    mother_occupation, father_occupation, displaced, debtor,
    tuition_fees_up_to_date, gender, scholarship_holder, age,
    avg_enrolled, avg_approved, avg_grade
]], columns=[
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime/evening_attendance', 'Previous_qualification',
    'Mother_qualification', 'Father_qualification', 'Mother_occupation',
    'Father_occupation', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
    'Gender', 'Scholarship_holder', 'Age', 'avg_enrolled',
    'avg_approved', 'avg_grade'
])

# --- Scale input ---
if 'scaler' in locals():
    try:
        scaled_input = scaler.transform(input_data)
        input_data = pd.DataFrame(scaled_input, columns=input_data.columns)
        st.success("📏 Input data scaled successfully.")
    except Exception as e:
        st.error(f"⚠️ Scaling error: {e}")
else:
    st.warning("Scaler not available. Input is not scaled.")

# --- Select true label for validation ---
# true_label = st.selectbox("True Label (for validation)", [0, 1], format_func=lambda x: "Dropout" if x == 0 else "Graduated")

# --- Predict button ---
if st.button("🔍 Predict"):
    if 'model' in locals():
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        label_map = {0: "⚠️ Dropout", 1: "🎓 Graduated"}
        result = label_map[prediction]

        st.subheader("📊 Prediction Result")
        st.success(f"The model predicts the student will **{result}**.")

        st.markdown(f"""
        **Prediction Confidence:**
        - Dropout: `{probabilities[0]:.2%}`
        - Graduated: `{probabilities[1]:.2%}`
        """)

    #     # --- Validate prediction ---
    #     true_label_text = label_map[true_label]
    #     is_correct = "✅ Correct" if prediction == true_label else "❌ Incorrect"

    #     st.subheader("📝 Validation")
    #     st.write(f"**Prediction:** {result}  \n**True Label:** {true_label_text}  \n**Validation Result:** {is_correct}")
    # else:
    #     st.error("❌ Model not loaded. Please check your model file.")
