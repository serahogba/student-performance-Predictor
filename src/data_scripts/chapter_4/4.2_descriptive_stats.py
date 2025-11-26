# file name: 4.2_descriptive_stats.py
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------------------------------------------
# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")

desc = student_df.describe(include="all").T
desc.to_excel("4.2_descriptive_summary.xlsx")
desc

# ----------------------------------------------------------------------------------------------
# 2. Socio-economic factors visualizations and their relation to Final Exam Scores
# ----------------------------------------------------------------------------------------------
# Distribution of Final Exam Scores
plt.figure(figsize=(6, 4))  # Set figure size
sns.histplot(student_df["Final_Exam_Score"], bins=20, kde=True)
plt.title("Distribution of Final Exam Scores")
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig("../../visualization/4.2_final_exam_hist.png", dpi=300)
plt.show()  # Display the plot
plt.close()  # Close the figure to free memory

# ----------------------------------------------------------------------------------------------

# Gender vs Pass/Fail Distribution
plt.figure(figsize=(6, 4))
sns.countplot(
    x="Gender",
    hue="Pass_Fail",  # hue for categorical separation
    data=student_df,
    palette="Set1",  # color palette for better display
)
plt.title("Gender vs Pass/Fail Distribution")
plt.tight_layout()
plt.savefig("../../visualization/4.2_gender_vs_pass_fail.png", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------------------------------------------------------

# Attendance vs final grade boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(
    x=pd.cut(student_df["Attendance_Rate"], bins=4), y=student_df["Final_Exam_Score"]
)  # Bin attendance rate into 4 categories
plt.title("Attendance vs Final Exam Score")
plt.savefig("../../visualization/4.2_attendance_vs_score.png", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------------------------------------------------------

# Study Hours vs Final Exam Score scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Study_Hours_per_Week",
    y="Final_Exam_Score",
    hue="Pass_Fail",
    data=student_df,
    palette="Set1",
)
plt.title("Study Hours vs Final Exam Score")
plt.savefig("../../visualization/4.2_study_hours_vs_score.png", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------------------------------------------------------

# Intenet Access vs Final Exam Score boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x="Internet_Access_at_Home", y="Final_Exam_Score", data=student_df)
plt.title("Internet Access vs Final Exam Score")
plt.savefig("../../visualization/4.2_internet_access_vs_score.png", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------------------------------------------------------

# Extracurricular Activities vs Final Exam Score boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x="Extracurricular_Activities", y="Final_Exam_Score", data=student_df)
plt.title("Extracurricular Activities vs Final Exam Score")
plt.savefig("../../visualization/4.2_extracurricular_vs_score.png", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------------------------------------------------------

# Past Exam Scores vs Final Exam Score scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Past_Exam_Scores",
    y="Final_Exam_Score",
    hue="Pass_Fail",
    data=student_df,
    palette="Set1",
)
plt.title("Past Exam Scores vs Final Exam Score")
plt.savefig("../../visualization/4.2_past_scores_vs_final.png", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------------------------------------------------------

# Parental Education Level vs Final Exam Score boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Parental_Education_Level", y="Final_Exam_Score", data=student_df)
plt.title("Parental Education Level vs Final Exam Score")
plt.savefig("../../visualization/4.2_parental_education_vs_score.png", dpi=300)
plt.show()
plt.close()
