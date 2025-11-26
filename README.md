## Data Science Project Template

You can use this template to structure your Python data science projects. It is based on [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

# 🎓 Student Performance Prediction using Machine Learning

## Overview
This project applies **supervised machine learning** to predict students’ academic performance using socio-economic features.  
The goal is to identify key factors influencing success and develop a deployable prediction model for early academic intervention and support.

The project uses a **publicly available Kaggle dataset** containing 708 student records, each with variables such as study hours, attendance, prior grades, and parental education level.

---

## Objectives
- AnalySe relationships between student attributes and final performance outcomes.  
- Develop and evaluate machine learning models (classification and regression).  
- Determine the most influential predictors of academic success.  
- Deploy a simple web app for real-time student performance prediction.

---

## Methodology Overview
### 1. Data Preparation
- Handled missing values (none detected).  
- Encoded categorical variables using **OneHotEncoder**.  
- Scaled numerical variables using **StandardScaler**.  

### 2. Exploratory Data Analysis
- Descriptive statistics and correlation heatmaps.  
- Outlier detection using boxplots.  
- Visual insights using Seaborn histograms and count plots.

### 3. Model Development
- **Classification (Pass/Fail):** Random Forest Classifier  
  - Accuracy: **~87%**
  - Balanced precision/recall across classes.  
- **Regression (Final Exam Score):** Random Forest Regressor  
  - MAE: **2.00**, RMSE: **2.91**, R²: **0.80**

### 4. Feature Importance & Explainability
Key predictors influencing student success:
- **Past Exam Scores**  
- **Attendance Rate**  
- **Study Hours per Week**

### 5. Hypothesis Testing
Significant correlations were observed between academic factors and Pass/Fail outcomes:
| Variable | Correlation (r) | p-value |
|-----------|----------------|---------|
| Study_Hours_per_Week | 0.297 | < 0.001 |
| Attendance_Rate | 0.396 | < 0.001 |
| Past_Exam_Scores | 0.437 | < 0.001 |

All factors showed statistically significant relationships (p < 0.05), confirming the research hypothesis.

---

## Application Deployment
The trained classification pipeline (`clf_pipeline_v1.joblib`) was deployed using **Streamlit**.

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run 4.10_app.py
