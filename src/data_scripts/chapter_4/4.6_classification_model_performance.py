# file name: 4.6_classification_model_performance.py
# Import necessary libraries
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ----------------------------------------------------------------------------------------------

# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")

# ----------------------------------------------------------------------------------------------
# Prepare data for classification model
X = student_df.drop(columns=["Final_Exam_Score", "Pass_Fail", "Student_ID"])  # Features
y = student_df["Pass_Fail"].map({"Fail": 0, "Pass": 1})  # Binary target variable

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
clf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),  # Preprocess the data
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)  # Stratify to maintain class balance

# Train the model
clf_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf_pipeline.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Save report to Excel and print
pd.DataFrame(report).T.to_excel("4.6_classification_report.xlsx")
print(pd.DataFrame(report).T)

# ----------------------------------------------------------------------------------------------
# Save the trained classification pipeline for deployment
joblib.dump(clf_pipeline, "../../../models/clf_pipeline_v1.joblib")
print("Model pipeline saved as clf_pipeline_v1.joblib")
