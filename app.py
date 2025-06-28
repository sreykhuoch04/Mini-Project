
import streamlit as st
import joblib
import pandas as pd

# --- Page config ---
st.set_page_config(page_title="🎓 Student Dropout Predictor", layout="centered")

st.markdown("<h1 style='text-align: center;'>🎓 Student Dropout Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Enter student information below to predict their dropout risk</h5>", unsafe_allow_html=True)
st.markdown("---")

# --- Load model and scaler ---
try:
    model = joblib.load("best_random_forest_model.plk")
    scaler = joblib.load("scaler.plk")
    st.success("✅ Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")

# --- Dictionaries ---
marital_status_map = {
    "Single": 1, "Married": 2, "Widowed": 3, "Divorced": 4,
    "Facto Union": 5, "Legally Separated": 6
}
application_mode_map = {
    "1st phase—general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase—special contingent (Azores Island)": 3,
    "Holders of other higher courses": 4,
    "Ordinance No. 854-B/99": 5,
    "International student (bachelor)": 6,
    "1st phase—special contingent (Madeira Island)": 7,
    "2nd phase—general contingent": 8,
    "3rd phase—general contingent": 9,
    "Ordinance No. 533-A/99, item b2) (Different Plan)": 10,
    "Ordinance No. 533-A/99, item b3 (Other Institution)": 11,
    "Over 23 years old": 12,
    "Transfer": 13,
    "Change in course": 14,
    "Technological specialization diploma holders": 15,
    "Change in institution/course": 16,
    "Short cycle diploma holders": 17,
    "Change in institution/course (International)": 18
}
course_map = {
    "Biofuel Production": 1, "Animation & Multimedia": 2, "Social Service": 3,
    "Agronomy": 4, "Communication Design": 5, "Veterinary Nursing": 6,
    "Informatics Engineering": 7, "Equinculture": 8, "Management": 9,
    "Informatics Management": 10, "Tourism": 11, "Nursing": 12,
    "Oral Hygiene": 13, "Marketing Management": 14,
    "Journalism": 15, "Basic Education": 16, "Management (Evening)": 17
}
qualification_map = {
    "Secondary education": 1,
    "Higher education—bachelor’s degree": 2,
    "Higher education—degree": 3,
    "Higher education—master’s degree": 4,
    "Higher education—doctorate": 5,
    "Frequency of higher education": 6,
    "12th year of schooling—not completed": 7,
    "11th year of schooling—not completed": 8,
    "Other—11th year of schooling": 9,
    "10th year of schooling": 10,
    "10th year of schooling—not completed": 11,
    "Basic education 3rd cycle (9th/10th/11th year) or equivalent": 12,
    "Basic education 2nd cycle (6th/7th/8th year) or equivalent": 13,
    "Technological specialization course": 14,
    "Higher education—degree (1st cycle)": 15,
    "Professional higher technical course": 16,
    "Higher education—master’s degree (2nd cycle)": 17
}
parent_qualification_map = {
    "Secondary Education—12th Year of Schooling or Equivalent": 1,
    "Higher Education—bachelor’s degree": 2,
    "Higher Education—degree": 3,
    "Higher Education—master’s degree": 4,
    "Higher Education—doctorate": 5,
    "Frequency of Higher Education": 6,
    "12th Year of Schooling—not completed": 7,
    "11th Year of Schooling—not completed": 8,
    "7th Year (Old)": 9,
    "Other—11th Year of Schooling": 10,
    "2nd year complementary high school course": 11,
    "10th Year of Schooling": 12,
    "General commerce course": 13,
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent": 14,
    "Complementary High School Course": 15,
    "Technical-professional course": 16,
    "Complementary High School Course—not concluded": 17,
    "7th year of schooling": 18,
    "2nd cycle of the general high school course": 19,
    "9th Year of Schooling—not completed": 20,
    "8th year of schooling": 21,
    "General Course of Administration and Commerce": 22,
    "Supplementary Accounting and Administration": 23,
    "Unknown": 24,
    "Cannot read or write": 25,
    "Can read without having a 4th year of schooling": 26,
    "Basic education 1st cycle (4th/5th year) or equivalent": 27,
    "Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent": 28,
    "Technological specialization course": 29,
    "Higher education—degree (1st cycle)": 30,
    "Specialized higher studies course": 31,
    "Professional higher technical course": 32,
    "Higher Education—master’s degree (2nd cycle)": 33,
    "Higher Education—doctorate (3rd cycle)": 34
}
parent_occupation_map = {
    "Student": 1,
    "Legislative/Executive Managers": 2,
    "Scientific Specialists": 3,
    "Intermediate Technicians": 4,
    "Administrative Staff": 5,
    "Service/Security/Sales": 6,
    "Farmers/Fishermen": 7,
    "Construction/Industry Workers": 8,
    "Machine Operators": 9,
    "Unskilled Workers": 10,
    "Armed Forces Professions": 11,
    "(Blank)": 13,
    "Other Situation": 12,
    "Armed Forces Officers": 14,
    "Armed Forces Sergeants": 15,
    "Other Armed Forces": 16,
    "Admin/Commercial Directors": 17,
    "Service Directors": 18,
    "Engineering/Math Specialists": 19,
    "Health Professionals": 20,
    "Teachers": 21,
    "Finance/Admin Specialists": 22,
    "Science Technicians": 23,
    "Health Technicians": 24,
    "Social/Sports Technicians": 25,
    "ICT Technicians": 26,
    "Secretaries/Data Ops": 27,
    "Accounting/Data Operators": 28,
    "Admin Support Staff": 29,
    "Personal Service Workers": 30,
    "Sellers": 31,
    "Care Workers": 32,
    "Security Services": 33,
    "Market Farmers": 34,
    "Subsistence Farmers/Fishermen": 35,
    "Construction Workers (not electricians)": 36,
    "Metal/Industry Workers": 37,
    "Electrical/Electronics Workers": 38,
    "Food/Wood/Textile Workers": 39,
    "Plant Operators": 40,
    "Assembly Workers": 41,
    "Drivers/Mobile Operators": 42,
    "Unskilled Agriculture Workers": 43,
    "Unskilled Industry/Transport": 44,
    "Meal Assistants": 45,
    "Street Vendors/Service": 46
}
attendance_map = {"Daytime": 1, "Evening": 0}
binary_map = {"No": 0, "Yes": 1}
gender_map = {"Male": 0, "Female": 1}

# --- Input form ---
with st.container():
    st.subheader("🎓 Personal Info")
    col1, col2 = st.columns(2)
    with col1:
        marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
        gender = st.radio("Gender", list(gender_map.keys()))
        age = st.number_input("Age", min_value=15, max_value=60, value=20)
        displaced = st.radio("Displaced", list(binary_map.keys()))
        debtor = st.radio("Debtor", list(binary_map.keys()))
        tuition_fees_up_to_date = st.radio("Tuition Fees Up-To-Date", list(binary_map.keys()))
        scholarship_holder = st.radio("Scholarship Holder", list(binary_map.keys()))

    with col2:
        mother_occupation = st.selectbox("Mother's Occupation", list(parent_occupation_map.keys()))
        father_occupation = st.selectbox("Father's Occupation", list(parent_occupation_map.keys()))
        mother_qualification = st.selectbox("Mother's Qualification", list(parent_qualification_map.keys()))
        father_qualification = st.selectbox("Father's Qualification", list(parent_qualification_map.keys()))
        previous_qualification = st.selectbox("Previous Qualification", list(qualification_map.keys()))

st.markdown("---")

with st.container():
    st.subheader("📚 Academic Background")
    col3, col4 = st.columns(2)
    with col3:
        application_mode = st.selectbox("Application Mode", list(application_mode_map.keys()))
        application_order = st.number_input("Application Order", min_value=1, value=1)
        course = st.selectbox("Course", list(course_map.keys()))
        attendance = st.radio("Attendance Type", list(attendance_map.keys()))

    with col4:
        avg_enrolled = st.number_input("Average Enrolled Credits", value=0.0)
        avg_approved = st.number_input("Average Approved Credits", value=0.0)
        avg_grade = st.number_input("Average Grade", value=0.0)

# --- Prepare DataFrame ---
input_data = pd.DataFrame([[ 
    marital_status_map[marital_status],
    application_mode_map[application_mode],
    application_order,
    course_map[course],
    attendance_map[attendance],
    qualification_map[previous_qualification],
    parent_qualification_map[mother_qualification],
    parent_qualification_map[father_qualification],
    parent_occupation_map[mother_occupation],
    parent_occupation_map[father_occupation],
    binary_map[displaced],
    binary_map[debtor],
    binary_map[tuition_fees_up_to_date],
    gender_map[gender],
    binary_map[scholarship_holder],
    age,
    avg_enrolled,
    avg_approved,
    avg_grade
]], columns=[
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime/evening_attendance', 'Previous_qualification',
    'Mother_qualification', 'Father_qualification', 'Mother_occupation',
    'Father_occupation', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
    'Gender', 'Scholarship_holder', 'Age', 'avg_enrolled',
    'avg_approved', 'avg_grade'
])

# --- Scale ---
if 'scaler' in locals():
    try:
        scaled_input = scaler.transform(input_data)
        input_data = pd.DataFrame(scaled_input, columns=input_data.columns)
    except Exception as e:
        st.error(f"⚠️ Scaling error: {e}")
else:
    st.warning("Scaler not available. Input not scaled.")

# --- Predict ---
if st.button("🔍 Predict"):
    if 'model' in locals():
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        result = "🎓 **Graduated**" if prediction == 1 else "⚠️ **Dropout**"
        st.markdown("### 🧠 Prediction Result")
        st.success(f"Model predicts the student will: {result}")

        st.markdown(f"""
        #### 📈 Prediction Confidence:
        - Dropout: `{prob[0]:.2%}`
        - Graduated: `{prob[1]:.2%}`
        """)
