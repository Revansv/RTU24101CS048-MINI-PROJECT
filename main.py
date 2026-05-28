import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Traffic Accident Analysis", layout="wide")

st.title("🚗 Traffic Accident Data Analysis")
st.write("Machine Learning Project using Random Forest Classifier")

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("US_Accidents_March23.csv")

# Select important columns
df = df[['Severity',
         'Temperature(F)',
         'Visibility(mi)',
         'Humidity(%)',
         'Wind_Speed(mph)']]

# Remove missing values
df.dropna(inplace=True)
df = df.head(100000)
# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# DATASET INFORMATION
# -------------------------------
st.subheader("Dataset Shape")
st.write("Rows and Columns:", df.shape)

# -------------------------------
# VISUALIZATION 1
# Severity Count Plot
# -------------------------------
st.subheader("Accident Severity Distribution")

fig1, ax1 = plt.subplots(figsize=(6,4))
sns.countplot(x=df['Severity'], ax=ax1)
st.pyplot(fig1)

# -------------------------------
# VISUALIZATION 2
# Correlation Heatmap
# -------------------------------
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# -------------------------------
# VISUALIZATION 3
# Temperature Distribution
# -------------------------------
st.subheader("Temperature Distribution")

fig3, ax3 = plt.subplots(figsize=(7,4))
sns.histplot(df['Temperature(F)'], bins=30, kde=True, ax=ax3)
st.pyplot(fig3)

# -------------------------------
# VISUALIZATION 4
# Visibility vs Severity
# -------------------------------
st.subheader("Visibility vs Severity")

fig4, ax4 = plt.subplots(figsize=(7,4))
sns.boxplot(x='Severity', y='Visibility(mi)', data=df, ax=ax4)
st.pyplot(fig4)

# -------------------------------
# MACHINE LEARNING
# -------------------------------

# Features and target
X = df.drop('Severity', axis=1)
y = df['Severity']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# MODEL RESULTS
# -------------------------------
st.subheader("Model Accuracy")

st.success(f"Accuracy: {accuracy:.2f}")

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig5, ax5 = plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig5)

# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------
st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# -------------------------------
# SIDEBAR PREDICTION
# -------------------------------
st.sidebar.header("Predict Accident Severity")

temp = st.sidebar.slider("Temperature(F)", -20.0, 120.0, 70.0)
visibility = st.sidebar.slider("Visibility(mi)", 0.0, 20.0, 10.0)
humidity = st.sidebar.slider("Humidity(%)", 0.0, 100.0, 50.0)
wind = st.sidebar.slider("Wind Speed(mph)", 0.0, 50.0, 5.0)

if st.sidebar.button("Predict Severity"):

    input_data = np.array([[temp, visibility, humidity, wind]])

    prediction = model.predict(input_data)

    st.sidebar.success(f"Predicted Severity: {prediction[0]}")
