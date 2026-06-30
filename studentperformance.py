#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import os
#dataloading
df = pd.read_excel("C:\\Users\\BHARATHWAJ\\OneDrive\\Desktop\\StudentsPerformance (1).xlsx")
print(df)
#dataprocessing
print(df.info)
print(df.head())
#data cleaning
print(df.tail())
print(df.describe())
print(df.isnull().sum())
print(df.shape)
#EDA
print("total of reading score:",df["reading score"].sum())  #sum
print("total of writing score:",df["writing score"].sum())
print("total of math score:",df["math score"].sum())

print("minimum of reading score:",df["reading score"].min()) #min
print("minimum of writing score:",df["writing score"].min())
print("minimum of math score:",df["math score"].min())

print("max of reading score:",df["reading score"].mode()) #max
print("max of writing score:",df["writing score"].mode())
print("max of math score:",df["math score"].mode())

print(df['math score'].median())

corr = df.corr(numeric_only=True)
print(corr)

#EDA graph
sns.barplot(x='gender',
            y='math score',
            data=df,
           color= 'red')
plt.title("test preparation impact")
plt.show()

sns.barplot(x='lunch',
           y='math score',
           data=df)
plt.show()

sns.histplot(df['math score'], kde=True, color= 'green')
plt.show()

sns.heatmap(corr, annot=True)
plt.show()

# Create Target Column

df["result"] = df["math score"].apply( lambda x: 1 if x >= 40 else 0)
print(df["result"].value_counts())


#encoding of string
label = LabelEncoder()

for col in [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]:
    df[col] =label.fit_transform(df[col])


#declare a x and y
x = df.drop('result',axis=1)
y = df['result']

#train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

#model buliding of logistics regression
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(lr_accuracy)
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test,lr_pred))

#decision tree classifier
dt = DecisionTreeClassifier(max_depth = 5,random_state=42)
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)
print(dt_pred)
dt_accuracy = accuracy_score(y_test,dt_pred)
print(dt_accuracy)
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test,dt_pred))

#KNN 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_test)
print(dt_pred)
knn_accuracy = accuracy_score(y_test,knn_pred)
print(knn_accuracy)
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test,knn_pred))

#model compare
models = ['Logistic', 'Decision Tree', 'KNN']

accuracy = [
    lr_accuracy,
    dt_accuracy,
    knn_accuracy
]

plt.figure(figsize=(8,5))

sns.barplot(
    x=models,
    y=accuracy
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

#plot for confusion matrix
plt.figure(figsize=(5,4))

sns.heatmap(
    confusion_matrix(y_test, dt_pred),
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")
plt.show()

#Feature Importance

importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': dt.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")
plt.show()

with open("lrml.pkl","wb") as f:
    pickle.dump(dt,f)

print("model saved..!")
print(os.path.exists("lrml.pkl"))

print("the student performance project successfully completed...!")

