# file name: 3.7_reliability_cv.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ----------------------------------------------------------------------------------------------
# Load the dataset, create label columns and encode categorical columns
# ----------------------------------------------------------------------------------------------
# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")

# Encode categorical columns using Label Encoding
cat_cols = student_df.select_dtypes(include=["object"]).columns
label_encoders = {}  # To store label encoders for each column
for col in cat_cols:
    le = LabelEncoder()  # Initialize label encoder
    student_df[col] = le.fit_transform(
        student_df[col].astype(str)
    )  # Fit and transform the column
    label_encoders[col] = le  # Store the encoder

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

# ----------------------------------------------------------------------------------------------

# Evaluate classification model reliability using Repeated Stratified K-Fold Cross-Validation
clf = RandomForestClassifier(n_estimators=200, random_state=42)  # Initialize classifier
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(
    clf, X_train, y_train_clf, cv=cv, scoring="f1"
)  # Compute F1 scores
print(
    "F1 mean: %.4f, std: %.4f" % (scores.mean(), scores.std())
)  # Print mean and std of F1 scores
