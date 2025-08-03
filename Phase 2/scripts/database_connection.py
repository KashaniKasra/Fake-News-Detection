import pandas as pd
import os
from sqlalchemy import create_engine

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "news.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)



# Create the directory structure for the database file
base_path_db = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
os.makedirs(base_path_db, exist_ok=True)
db_file = os.path.join(base_path_db, "news_dataset.db")

# Connect to SQLite database (or create it if it doesn't exist)
engine = create_engine(f"sqlite:///{db_file}")

# Write the dataframe to a SQL table named 'news'
df.to_sql("news", con=engine, index=False, if_exists="replace")