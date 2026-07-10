import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    with open("lrml.pkl", "rb") as f:
        return pickle.load(f)

try:
    lr = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'lrml.pkl' not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🎓 STUDENT PERFORMANCE ANALYSIS")
st.write("### Machine Learning Based Student Pass/Fail Prediction")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("Student Details")
st.sidebar.markdown("---")

gender = st.sidebar.radio(
    "Gender",
    ["male", "female"]
)

race_ethnicity = st.sidebar.selectbox(
    "Race Ethnicity",
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
)

parental_level_of_education = st.sidebar.selectbox(
    "Parental Education",
    [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school"
    ]
)

lunch_type = st.sidebar.radio(
    "Lunch Type",
    [
        "free/reduced",
        "standard"
    ]
)

test_preparation_course = st.sidebar.radio(
    "Test Preparation",
    [
        "none",
        "completed"
    ]
)

st.sidebar.subheader("Subject Marks")

math_score = st.sidebar.slider("Math Score", 0, 100, 40)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 40)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 40)

# -------------------------------------------------
# Encoding
# -------------------------------------------------
gender_value = 1 if gender == "male" else 0

race_map = {
    "group A": 0,
    "group B": 1,
    "group C": 2,
    "group D": 3,
    "group E": 4
}

parent_map = {
    "associate's degree": 0,
    "bachelor's degree": 1,
    "high school": 2,
    "master's degree": 3,
    "some college": 4,
    "some high school": 5
}

input_data = pd.DataFrame({
    "gender": [gender_value],
    "race/ethnicity": [race_map[race_ethnicity]],
    "parental level of education": [
        parent_map[parental_level_of_education]
    ],
    "lunch": [
        1 if lunch_type == "standard" else 0
    ],
    "test preparation course": [
        1 if test_preparation_course == "completed" else 0
    ],
    "math score": [math_score],
    "reading score": [reading_score],
    "writing score": [writing_score]
})

# -------------------------------------------------
# Input Summary
# -------------------------------------------------
st.subheader("Student Input Summary")

display_data = pd.DataFrame({
    "Gender": [gender],
    "Race": [race_ethnicity],
    "Parental Education": [parental_level_of_education],
    "Lunch": [lunch_type],
    "Test Preparation": [test_preparation_course],
    "Math": [math_score],
    "Reading": [reading_score],
    "Writing": [writing_score]
})

st.dataframe(display_data, use_container_width=True)

average = (math_score + reading_score + writing_score) / 3

st.metric(
    "Average Marks",
    f"{average:.2f}"
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🚀 Predict Performance"):

    prediction = lr.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

    if hasattr(lr, "predict_proba"):

        probability = lr.predict_proba(input_data)[0][1]

        st.metric(
            "Pass Probability",
            f"{probability*100:.2f}%"
        )

        if probability >= 0.80:
            st.success("Confidence : Very High")

        elif probability >= 0.60:
            st.info("Confidence : Moderate")

        else:
            st.warning("Confidence : Low")

    report = display_data.copy()
    report["Average"] = average
    report["Prediction"] = (
        "PASS"
        if prediction[0] == 1
        else "FAIL"
    )

    csv = report.to_csv(index=False)

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "student_report.csv",
        "text/csv"
    )

# -------------------------------------------------
# Graph Data
# -------------------------------------------------
graph_data = pd.DataFrame({
    "Subject": [
        "Math",
        "Reading",
        "Writing"
    ],
    "Marks": [
        math_score,
        reading_score,
        writing_score
    ]
})

# -------------------------------------------------
# Plotly Bar Chart
# -------------------------------------------------
st.subheader("📊 Student Performance")

fig = px.bar(
    graph_data,
    x="Subject",
    y="Marks",
    color="Subject",
    text="Marks",
    title="Subject Wise Marks"
)

fig.update_layout(yaxis_range=[0, 100])

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------------------------
# Matplotlib Chart
# -------------------------------------------------
fig2, ax = plt.subplots(figsize=(6, 4))

ax.bar(
    graph_data["Subject"],
    graph_data["Marks"],
    color=[
        "skyblue",
        "lightgreen",
        "orange"
    ]
)

ax.axhline(
    y=40,
    color="red",
    linestyle="--",
    label="Pass Mark"
)

ax.set_ylim(0, 100)
ax.set_ylabel("Marks")
ax.legend()

st.pyplot(fig2)

# -------------------------------------------------
# Pie Chart
# -------------------------------------------------
pie = px.pie(
    graph_data,
    values="Marks",
    names="Subject",
    title="Marks Distribution"
)

st.plotly_chart(
    pie,
    use_container_width=True
)

# -------------------------------------------------
# Analysis
# -------------------------------------------------
highest = graph_data.loc[
    graph_data["Marks"].idxmax()
]

lowest = graph_data.loc[
    graph_data["Marks"].idxmin()
]

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"Highest Score : {highest['Subject']} ({highest['Marks']})"
    )

with col2:
    st.warning(
        f"Lowest Score : {lowest['Subject']} ({lowest['Marks']})"
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")

st.caption(
    f"Prediction Time : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)

st.markdown("""
## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- Logistic Regression
- Machine Learning
- Plotly
- Matplotlib
- Pickle
""")
