
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Health Insurance Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# 2. LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("random_forest_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_files()
except Exception as e:
    st.error(f"Unable to load model or preprocessor: {e}")
    st.stop()

# --------------------------------------------------
# 3. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .header-box {
        background: linear-gradient(120deg, #12345a, #247ba0);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
    }

    .header-box h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 8px;
    }

    .header-box p {
        color: #e8f2fa;
        font-size: 16px;
    }

    div.stButton > button {
        background-color: #176b87;
        color: white;
        border-radius: 9px;
        height: 48px;
        font-size: 17px;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #104d65;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: #edf6f9;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #d5e8ee;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.markdown("""
<div class="header-box">
    <h1>🏥 Health Insurance Claim Prediction</h1>
    <p>
        Estimate health insurance claim amount using
        machine learning based on personal and health details.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 5. CITY AND JOB LISTS
# --------------------------------------------------

cities = [
    'NewYork', 'Boston', 'Phildelphia', 'Pittsburg',
    'Buffalo', 'AtlanticCity', 'Portland', 'Cambridge',
    'Hartford', 'Springfield', 'Syracuse', 'Baltimore',
    'York', 'Trenton', 'Warwick', 'WashingtonDC',
    'Providence', 'Harrisburg', 'Newport', 'Stamford',
    'Worcester', 'Atlanta', 'Brimingham', 'Charleston',
    'Charlotte', 'Louisville', 'Memphis', 'Nashville',
    'NewOrleans', 'Raleigh', 'Houston', 'Georgia',
    'Oklahoma', 'Orlando', 'Macon', 'Huntsville',
    'Knoxville', 'Florence', 'Miami', 'Tampa',
    'PanamaCity', 'Kingsport', 'Marshall', 'Mandan',
    'Waterloo', 'IowaCity', 'Columbia', 'Indianapolis',
    'Cincinnati', 'Bloomington', 'Salina', 'KanasCity',
    'Brookings', 'Minot', 'Chicago', 'Lincoln',
    'FallsCity', 'GrandForks', 'Fargo', 'Cleveland',
    'Canton', 'Columbus', 'Rochester', 'Minneapolis',
    'JeffersonCity', 'Escabana', 'Youngstown',
    'SantaRosa', 'Eureka', 'SanFrancisco', 'SanJose',
    'LosAngeles', 'Oxnard', 'SanDeigo', 'Oceanside',
    'Carlsbad', 'Montrose', 'Prescott', 'Fresno',
    'Reno', 'LasVegas', 'Tucson', 'SanLuis',
    'Denver', 'Kingman', 'Bakersfield', 'Mexicali',
    'SilverCity', 'Pheonix', 'SantaFe', 'Lovelock'
]

job_titles = [
    'Actor', 'Engineer', 'Academician', 'Chef',
    'HomeMakers', 'Dancer', 'Singer', 'DataScientist',
    'Police', 'Student', 'Doctor', 'Manager',
    'Photographer', 'Beautician', 'CA', 'Blogger',
    'CEO', 'Labourer', 'Accountant', 'FilmDirector',
    'Technician', 'FashionDesigner', 'Architect',
    'HouseKeeper', 'FilmMaker', 'Buisnessman',
    'Politician', 'DefencePersonnels', 'Analyst',
    'Clerks', 'ITProfessional', 'Farmer',
    'Journalist', 'Lawyer', 'GovEmployee'
]

diseases = [
    'NoDisease', 'Epilepsy', 'EyeDisease', 'Alzheimer',
    'Arthritis', 'HeartDisease', 'Diabetes', 'Cancer',
    'High BP', 'Obesity'
]

# --------------------------------------------------
# 6. INPUT FORM
# --------------------------------------------------

st.subheader("📋 Enter Your Details")
st.caption("Please provide the following information to estimate your claim.")

with st.form("insurance_form"):

    # PERSONAL DETAILS
    st.markdown("### 👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=1.0,
            max_value=300.0,
            value=65.0,
            step=0.5
        )

    with col2:
        bmi = st.number_input(
            "BMI",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=0
        )

        city = st.selectbox(
            "City",
            cities
        )

    st.divider()

    # HEALTH DETAILS
    st.markdown("### 🩺 Health Information")

    col3, col4 = st.columns(2)

    with col3:
        hereditary_diseases = st.selectbox(
            "Hereditary Diseases",
            diseases
        )

        bloodpressure = st.number_input(
            "Blood Pressure",
            min_value=1,
            max_value=250,
            value=80
        )

    with col4:
        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"]
        )

        smoker = st.selectbox(
            "Smoker",
            ["No", "Yes"]
        )

    st.divider()

    # LIFESTYLE AND OCCUPATION
    st.markdown("### 🏃 Lifestyle & Occupation")

    col5, col6 = st.columns(2)

    with col5:
        regular_ex = st.selectbox(
            "Regular Exercise",
            ["No", "Yes"]
        )

    with col6:
        job_title = st.selectbox(
            "Job Title",
            job_titles
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Insurance Claim",
        use_container_width=True
    )

# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "weight": weight,
        "bmi": bmi,
        "hereditary_diseases": hereditary_diseases,
        "no_of_dependents": no_of_dependents,
        "smoker": 1 if smoker == "Yes" else 0,
        "city": city,
        "bloodpressure": bloodpressure,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "regular_ex": 1 if regular_ex == "Yes" else 0,
        "job_title": job_title
    }])

    try:
        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)[0]

        st.divider()

        st.subheader("📊 Prediction Result")

        st.success("Insurance claim prediction completed!")

        st.metric(
            label="Estimated Insurance Claim Amount",
            value=f"{prediction:,.2f}"
        )

        st.info(
            "Note: This amount is an estimated prediction "
            "generated by a machine learning model. "
            "It is not a guaranteed insurance claim amount."
        )

        with st.expander("View Submitted Information"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning(
            "Please verify that all input categories "
            "match the values used during model training."
        )

# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Health Insurance Claim Prediction | "
    "Machine Learning Mini Project"
)
