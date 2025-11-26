# file name: 4.4_outlier_check.py
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------------------------
# Load the dataset
# ----------------------------------------------------------------------------------------------
# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")

# ----------------------------------------------------------------------------------------------

# Outlier detection using boxplots for key numeric variables
for col in ["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores"]:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=student_df[col], color="skyblue")
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"../../visualization/4.4_boxplot_{col}.png", dpi=300)
    plt.show()
    plt.close()
