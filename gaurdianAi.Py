
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Guardian AI",
    page_icon="🎓",
    layout="wide"
)

model = joblib.load("guardian_ai_xgboost.pkl")

st.title("🎓 Guardian AI")
st.subheader("Student Dropout Risk Prediction System")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 5, 25, 15)
    gender = st.selectbox("Gender", ["Male", "Female"])
    attendance_rate_pct = st.slider("Attendance (%)", 0.0, 100.0, 80.0)
    average_test_score_pct = st.slider("Average Test Score (%)", 0.0, 100.0, 70.0)
    household_income_monthly_usd = st.number_input("Monthly Household Income", 0.0, 100000.0, 5000.0)
    parental_support = st.selectbox("Parental Support", ["Yes", "No"])
    household_has_internet_access = st.selectbox("Internet Access", ["Yes", "No"])

with col2:
    receives_midday = st.selectbox("Receives Midday Meal", ["Yes", "No"])
    has_disability = st.selectbox("Disability", ["Yes", "No"])
    study_hours_per_day = st.number_input("Study Hours Per Day", 0.0, 15.0, 2.0)
    extracurricular_participation = st.selectbox("Extracurricular Participation", ["Yes", "No"])
    parent_education_level = st.selectbox(
        "Parent Education",
        ["Primary", "Secondary", "Higher Secondary", "Graduate", "Postgraduate"]
    )
    school_type = st.selectbox(
        "School Type",
        ["Government", "Private", "Aided"]
    )
    location = st.selectbox(
        "Location",
        ["Urban", "Semi-Urban", "Rural"]
    )

gender = 1 if gender == "Male" else 0

yes_no = {
    "Yes": 1,
    "No": 0
}

parent_map = {
    "Primary":0,
    "Secondary":1,
    "Higher Secondary":2,
    "Graduate":3,
    "Postgraduate":4
}

school_map = {
    "Government":0,
    "Private":1,
    "Aided":2
}

location_map = {
    "Urban":0,
    "Semi-Urban":1,
    "Rural":2
}

if st.button("Predict Risk Level", use_container_width=True):

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "attendance_rate_pct": attendance_rate_pct,
        "average_test_score_pct": average_test_score_pct,
        "household_income_monthly_usd": household_income_monthly_usd,
        "parental_support": yes_no[parental_support],
        "household_has_internet_access": yes_no[household_has_internet_access],
        "receives_midday": yes_no[receives_midday],
        "has_disability": yes_no[has_disability],
        "study_hours_per_day": study_hours_per_day,
        "extracurricular_participation": yes_no[extracurricular_participation],
        "parent_education_level": parent_map[parent_education_level],
        "school_type": school_map[school_type],
        "location": location_map[location]
    }])

    prediction = model.predict(input_data)[0]

    st.markdown("---")

    if prediction == 0:
        st.success("🟢 Low Dropout Risk")
        st.info("""
        **Recommendations**
        - Continue regular monitoring.
        - Encourage extracurricular activities.
        - Maintain attendance above 90%.
        """)

    elif prediction == 1:
        st.warning("🟡 Medium Dropout Risk")
        st.info("""
        **Recommendations**
        - Improve attendance.
        - Provide academic mentoring.
        - Increase parental involvement.
        """)

    elif prediction == 2:
        st.error("🟠 High Dropout Risk")
        st.info("""
        **Recommendations**
        - Immediate counselling.
        - Weekly academic monitoring.
        - Financial assistance if required.
        """)

    else:
        st.error("🔴 Critical Dropout Risk")
        st.info("""
        **Recommendations**
        - Emergency intervention.
        - Assign dedicated mentor.
        - Home visit and counselling.
        - Continuous performance monitoring.
        """)
