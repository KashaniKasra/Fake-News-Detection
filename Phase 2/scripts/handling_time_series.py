import pandas as pd
import os

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "temp3.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)
os.remove(data_file)



# Convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Extract day from the date
df["publish_dayofweek"] = df["date"].dt.dayofweek # 0 = Monday, ..., 6 = Sunday

# Extract month from the date
df["publish_month"] = df["date"].dt.month

# Specify the news that was published on weekends 
df["is_weekend"] = df["publish_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

# Extract season from the date
def get_season(month):  # 1 = Winter 2 = Spring, 3 = Summer, 4 = Autumn
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    elif month in [9, 10, 11]:
        return 4
    else:
        return 0

df["season"] = df["publish_month"].apply(get_season)



# Save the updated dataframe back to the CSV file
data_file = os.path.join(base_path_data, "temp4.csv")
df.to_csv(data_file, index=False)