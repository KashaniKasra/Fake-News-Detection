import pandas as pd
import os

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "temp2.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)
os.remove(data_file)



# Label encoding
df["label"] = df["label"].map({"fake": 0, "true": 1})

# One-hot encoding for general_category and subject
df = pd.get_dummies(df, columns=["general_category", "subject"], drop_first=True)
df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)



# Save the updated dataframe back to the CSV file
data_file = os.path.join(base_path_data, "temp3.csv")
df.to_csv(data_file, index=False)