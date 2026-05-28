# 🚗 Traffic Accident Data Analysis

A Machine Learning based web application built using **Streamlit**, **Pandas**, **Seaborn**, and **Scikit-learn** to analyze traffic accident data and predict accident severity using a **Random Forest Classifier**.

---

# 📌 Project Overview

This project performs:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Data Visualization
* Machine Learning Model Training
* Accident Severity Prediction

The application uses accident-related environmental conditions like:

* Temperature
* Visibility
* Humidity
* Wind Speed

to predict the severity of accidents.

---

# 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# 📂 Dataset

Dataset Used:

`US_Accidents_March23.csv`

Important columns used in this project:

* Severity
* Temperature(F)
* Visibility(mi)
* Humidity(%)
* Wind_Speed(mph)

---

# 📊 Features of the Project

## ✅ Dataset Preview

Displays the first few rows of the dataset.

## ✅ Data Visualizations

### 1. Accident Severity Distribution

Shows count of different accident severity levels.

### 2. Correlation Heatmap

Displays relationships between features.

### 3. Temperature Distribution

Histogram showing temperature distribution.

### 4. Visibility vs Severity

Boxplot comparing visibility with accident severity.

---

# 🤖 Machine Learning Model

## Algorithm Used

Random Forest Classifier

## Steps Performed

1. Data Preprocessing
2. Train-Test Split
3. Model Training
4. Prediction
5. Accuracy Evaluation

---

# 📈 Model Evaluation

The following evaluation metrics are used:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

# 🎯 Accident Severity Prediction

Users can predict accident severity using sidebar sliders:

* Temperature(F)
* Visibility(mi)
* Humidity(%)
* Wind Speed(mph)

After entering values, click:

`Predict Severity`

to get the predicted accident severity.

---

# ▶️ How to Run the Project

## Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

## Step 2: Place Dataset

Keep the dataset file:

```bash
US_Accidents_March23.csv
```

inside the project folder.

---

## Step 3: Run Streamlit App

```bash
streamlit run app.py
```

---

# 📷 Output Screens

The application displays:

* Dataset Preview
* Graphs and Charts
* Heatmaps
* Confusion Matrix
* Model Accuracy
* Prediction Section

---

# 📌 Future Improvements

* Add more features for prediction
* Improve model accuracy
* Deploy project online
* Add real-time accident analysis
* Use Deep Learning models

---

# 👨‍💻 Author

Revanasidda.

BTech CSE-AIML Student
Rai Technology University

---
