import pandas as pd
import os
import sqlite3

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "final_news.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)



# Create the directory structure for the database file
base_path_query = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
os.makedirs(base_path_query, exist_ok=True)
db_file = os.path.join(base_path_query, "news_dataset.db")

# Connect to the SQLite database
conn = sqlite3.connect(db_file)

# Write the final dataframe to a SQL table named 'final_news'
df.to_sql("final_news", conn, index=False, if_exists="replace")