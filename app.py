import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# Load Model
with open(r"C:\Users\BHARATHWAJ\lrml.pkl", "rb") as f:
    lr = pickle.load(f)


# Page Settings
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)


# Title
st.title("🎓 STUDENT PERFORMANCE ANALYSIS")
st.write("Machine Learning Based Student Pass/Fail Prediction")


# Sidebar Inputs
st.sidebar.header("Enter Student Details")


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



# Marks
st.sidebar.subheader("Subject Marks")


math_score = st.sidebar.slider(
    "Math Score",
    0,
    100,
    40
)


reading_score = st.sidebar.slider(
    "Reading Score",
    0,
    100,
    40
)


writing_score = st.sidebar.slider(
    "Writing Score",
    0,
    100,
    40
)



# Encoding

gender = 1 if gender == "male" else 0


race_map = {
    "group A":0,
    "group B":1,
    "group C":2,
    "group D":3,
    "group E":4
}

race_ethnicity = race_map[race_ethnicity]


parent_map = {
    "associate's degree":0,
    "bachelor's degree":1,
    "high school":2,
    "master's degree":3,
    "some college":4,
    "some high school":5
}

parental_level_of_education = parent_map[
    parental_level_of_education
]


lunch_type = 1 if lunch_type == "standard" else 0


test_preparation_course = (
    1 if test_preparation_course == "completed"
    else 0
)



# Prediction

if st.button("🚀 Predict Performance"):


    input_data = pd.DataFrame({

        "gender":[gender],

        "race/ethnicity":[race_ethnicity],

        "parental level of education":
        [parental_level_of_education],

        "lunch":[lunch_type],

        "test preparation course":
        [test_preparation_course],

        "math score":[math_score],

        "reading score":[reading_score],

        "writing score":[writing_score]

    })


    prediction = lr.predict(input_data)


    st.subheader("Prediction Result")


    if prediction[0] == 1:

        st.success(
            "✅ Student Performance: PASS"
        )

    else:

        st.error(
            "❌ Student Performance: FAIL"
        )


    # Probability
    if hasattr(lr,"predict_proba"):

        result = lr.predict_proba(input_data)

        st.metric(
            "Pass Probability",
            f"{result[0][1]*100:.2f}%"
        )



# Graph Section

st.subheader("📊 Performance Visualization")


graph_data = pd.DataFrame({

    "Subject":
    [
        "Math",
        "Reading",
        "Writing"
    ],

    "Marks":
    [
        math_score,
        reading_score,
        writing_score
    ]

})


fig = px.bar(
    graph_data,
    x="Subject",
    y="Marks",
    color="Subject",
    title="Student Marks"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# Pass mark chart

fig,ax = plt.subplots()


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


ax.set_ylim(0,100)

ax.legend()


st.pyplot(fig)



st.markdown(
"""
---
### Technologies Used
- Python
- Machine Learning
- Logistic Regression
- Streamlit
- Pandas
- Data Visualization
"""
)
