# file name: 4.9_hypothesis_testing.py
# Import necessary libraries
import pandas as pd
from scipy.stats import pearsonr

# ----------------------------------------------------------------------------------------------

# Load the dataset
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")

# ----------------------------------------------------------------------------------------------
# Encode Pass/Fail into numeric form for correlation testing
student_df["Pass_Fail_Binary"] = student_df["Pass_Fail"].map({"Fail": 0, "Pass": 1})

# Select academic factors for testing
factors = ["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores"]

# Compute Pearson correlation with Pass/Fail outcome
results = []  # Initialize results holder
for col in factors:  # Iterate through factors to test
    r, p = pearsonr(
        student_df[col], student_df["Pass_Fail_Binary"]
    )  # Pearson correlation
    results.append(
        {"Variable": col, "Correlation (r)": r, "p-value": p}
    )  # Store results in list

correlation_df = pd.DataFrame(results)  # Convert results to DataFrame
correlation_df.to_excel(
    "4.9_hypothesis_test_results.xlsx", index=False
)  # Save to Excel file
print(correlation_df)
