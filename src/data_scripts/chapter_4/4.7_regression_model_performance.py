# file name: 4.7_regression_model_performance.py
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------------------------

# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")

# ----------------------------------------------------------------------------------------------
# Prepare data for regression model
X = student_df.drop(columns=["Final_Exam_Score", "Pass_Fail", "Student_ID"])  # Features
y = student_df["Final_Exam_Score"]  # Continuous target variable

# Identify categorical and numerical columns
categorical_cols = [
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities",
]
numerical_cols = ["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores"]

# ----------------------------------------------------------------------------------------------
# Preprocessing pipeline for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),  # Standardize numerical features
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ),  # One-hot encode categorical features
    ]
)

# ----------------------------------------------------------------------------------------------
# Build model pipeline
reg_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),  # Preprocess the data
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
    ]
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # No stratification for regression to allow continuous target variable

# Train the model
reg_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg_pipeline.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared

# Save metrics to Excel and print results
results_df = pd.DataFrame({"Metric": ["MAE", "RMSE", "R²"], "Value": [mae, rmse, r2]})
results_df.to_excel("4.7_regression_metrics.xlsx", index=False)
print(results_df)
