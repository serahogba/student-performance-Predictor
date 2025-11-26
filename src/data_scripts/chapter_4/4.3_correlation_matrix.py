# file name: 4.3__correlation_matrix.py
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------------------------
# Load the dataset
# ----------------------------------------------------------------------------------------------
# Load the raw data into a pandas DataFrame
student_df = pd.read_csv("../../../data/raw/student_performance_dataset.csv")
# Calculate the correlation matrix for numeric variables
corr = student_df.corr(numeric_only=True)

# ----------------------------------------------------------------------------------------------

# Plot the correlation matrix
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Matrix of Numeric Variables")
plt.tight_layout()
plt.savefig("../../visualization/4.3_corr_matrix.png", dpi=300)
plt.show()
plt.close()
