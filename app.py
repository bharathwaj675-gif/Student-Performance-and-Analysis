import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open(r"C:\Users\BHARATHWAJ\lrml.pkl", "rb") as f:
    lr = pickle.load(f)

st.title("STUDENT PERFORMANCE ANALYSIS")  #bulid the app

gender = st.radio("Gender", ["male", "female"])

race_ethnicity = st.selectbox(
    "Race Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

parental_level_of_education = st.selectbox(
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

lunch_type = st.radio(
    "Lunch Type",
    ["free/reduced", "standard"]
)

test_preparation_course = st.radio(
    "Test Preparation Course",
    ["none", "completed"]
)

math_score = st.number_input("Math Score", 0 ,100,40)

reading_score = st.number_input("Reading Score", 0, 100,40)

writing_score = st.number_input("Writing Score", 0, 100, 40)


if st.button("Predict"):

    gender = 1 if gender == "male" else 0
    
    race_map = {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}
    race_ethnicity = race_map[race_ethnicity]

    parent_map = {"associate's degree": 0, "bachelor's degree": 1, 'high school': 2, "master's degree": 3, 'some college': 4, 'some high school': 5}
    parental_level_of_education = parent_map[parental_level_of_education]

    lunch_type = 1 if lunch_type == "standard" else 0

    test_preparation_course = 1 if test_preparation_course == "completed" else 0
     
    
    input_data = pd.DataFrame({
        "gender":[gender],
        "race/ethnicity":[race_ethnicity],
        "parental level of education":[parental_level_of_education],
        "lunch":[lunch_type],
        "test preparation course":[test_preparation_course],
        "math score":[math_score],
        "reading score":[reading_score],
        "writing score":[writing_score]
    })

    prediction = lr.predict(input_data)
    if prediction[0] == 1:
      st.success("pass")
    else:
      st.error("fail")

if st.button("show graphs"):
    subject = ['math_score', 'reading_score', 'writing_score']
    marks = [math_score, reading_score, writing_score]
    fig, ax = plt.subplots()
    ax.bar(subject, marks)
    ax.axhline(y=40, linestyle = '-', label = 'pass mark(40)')
    ax.set_ylim(0,100)
    st.pyplot(fig)

