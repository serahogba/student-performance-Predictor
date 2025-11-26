# file anme: 3.4_train_test_split.py
# Import the neccessary library
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------------------

# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")

# ----------------------------------------------------------------------------------------------

# Split the dataset into training and testing sets
X = student_df.drop(columns=["Final_Exam_Score", "Pass_Fail", "Student_ID"])
y_clf = student_df["Pass_Fail"]  # Classification target
y_reg = student_df["Final_Exam_Score"]  # Regression target

X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, stratify=y_clf, random_state=42
)  # Stratify for classification to maintain class balance
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)  # No stratification for regression to allow continuous target variable
# Print the shapes of the resulting datasets
print("Classification train/test:", X_train.shape, X_test.shape)
print("Regression train/test:", X_train_reg.shape, X_test_reg.shape)
