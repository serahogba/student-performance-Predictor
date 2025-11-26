# file name: 3.3_data_overview.py
# Import the neccessary library
import pandas as pd

# ----------------------------------------------------------------------------------------------
# 1. Load the dataset and create a variable summary
# ----------------------------------------------------------------------------------------------
# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")
summary = pd.DataFrame(
    {
        "dtype": student_df.dtypes.astype(
            str
        ),  # Convert dtypes to string for better readability
        "n_missing": student_df.isna().sum(),  # Count of missing values
        "pct_missing": student_df.isna().mean().round(2)
        * 100,  # Percentage of missing values
        "n_unique": student_df.nunique(),  # Number of unique values
    }
)  # Create initial summary DataFrame
desc = student_df.describe(
    include="all"
).T  # Get descriptive statistics for all columns
summary = summary.join(
    desc[["mean", "std", "min", "25%", "50%", "75%", "max"]], how="left"
)  # Join descriptive statistics  to the summary

# Move the index (variable names) into a column called 'Columns'
summary = summary.reset_index().rename(columns={"index": "Columns"})

# ----------------------------------------------------------------------------------------------
# 2. Export the Variable Summary to Excel Format
# ----------------------------------------------------------------------------------------------
# Save to Excel file without the index
summary.to_excel("3.3_variable_summary.xlsx", index=False)
print("Variable summary saved to 3.3_variable_summary.xlsx")

# Read the excel file to verify
student_df_excel = pd.read_excel("3.3_variable_summary.xlsx")
# Display the dataframe
student_df_excel
