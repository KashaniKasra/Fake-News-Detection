import pandas as pd
import numpy as np
import os

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "temp6.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)
os.remove(data_file)



# Remove irrelevant columns
df.drop(columns=["title", "text", "date", "subject_Middle-east", "subject_News", "subject_US_News",
                 "subject_left-news", "subject_politics", "subject_politicsNews", "subject_worldnews"], inplace=True)

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Remove columns with low variance (only one unique value)
low_variance = []

for col in df.columns:
    if not isinstance(df[col].iloc[0], (np.ndarray, list)):
        if df[col].nunique() == 1:
            low_variance.append(col)

df.drop(columns=low_variance, inplace=True)

# Remove columns with high sparsity (more than 98% zeros) except binary columns
sparse_cols = []

for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        if (df[col] == 0).sum() / len(df) > 0.98 and col not in ["lable", "is_weekend", "is_question_title"]:
            sparse_cols.append(col)

df.drop(columns=sparse_cols, inplace=True)

# Specify the numeric columns to keep for correlation analysis
numeric_df = df.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr_matrix = df.corr().abs()

# Set the threshold for correlation
threshold = 0.90

# Keep track of columns to drop
to_drop = set()

# List of columns in the correlation matrix
columns = corr_matrix.columns

# Delete the second column of the correlation matrix that have a high correlation with the first column
for i in range(len(columns)):
    if columns[i] in to_drop:
        continue
    for j in range(i + 1, len(columns)):
        if columns[j] in to_drop:
            continue
        if corr_matrix.iloc[i, j] > threshold:
            to_drop.add(columns[j])

# Drop the selected columns from the dataframe
df = df.drop(columns=list(to_drop))



# Save the updated dataframe back to the CSV file
data_file = os.path.join(base_path_data, "final_news.csv")
df.to_csv(data_file, index=False)