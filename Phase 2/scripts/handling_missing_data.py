import numpy as np
import pandas as pd
import os

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "temp4.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)
os.remove(data_file)



# Find missing values in the dataframe
missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

# Fill NaN values in the date column with a default date
df["date"] = df["date"].fillna(pd.to_datetime("2000-01-01"))

# Find NaN indices in day column
nan_indices_day = df[df["publish_dayofweek"].isna()].index

# Total number of NaNs
n_missing_day = len(nan_indices_day)

# Days of week (0 to 6)
days = list(range(7))

# Repeat days enough times and slice exactly n_missing elements
repeated_days = (days * ((n_missing_day // 7) + 1))[:n_missing_day]

# Shuffle for better distribution
np.random.shuffle(repeated_days)

# Fill NaNs with the distributed days
df.loc[nan_indices_day, "publish_dayofweek"] = repeated_days

# Find NaN indices in month column
nan_indices_month = df[df["publish_month"].isna()].index

# Total number of NaNs
n_missing_month = len(nan_indices_month)

# Months (1 to 12)
months = list(range(1, 13))

# Repeat months enough times and slice exactly n_missing elements
repeated_months = (months * ((n_missing_month // 12) + 1))[:n_missing_month]

# Shuffle for better distribution
np.random.shuffle(repeated_months)

# Fill NaNs with the distributed months
df.loc[nan_indices_month, "publish_month"] = repeated_months

# Check missing values in the dataframe again
missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)



# Save the updated dataframe back to the CSV file
data_file = os.path.join(base_path_data, "temp5.csv")
df.to_csv(data_file, index=False)