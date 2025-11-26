# file name: 3.6_pilot_eda.py
# Import the neccessary library
import pandas as pd

# ----------------------------------------------------------------------------------------------

# Load the dataset and create a pilot EDA summary
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")
# Create a pilot sample (10% of the data)
pilot = student_df.sample(frac=0.1, random_state=42)
# Save pilot descriptive statistics and missing value summarys
pilot.describe(include="all").T.to_excel("3.6_pilot_summary.xlsx")
pilot.isna().sum().to_excel("3.6_pilot_missing.xlsx")
# Print confirmation message
print("Pilot EDA outputs saved.")
