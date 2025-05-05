import numpy as np
import pandas as pd
import sqlite3
import os

# Create the directory structure for the database file
base_path_query = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
os.makedirs(base_path_query, exist_ok=True)
db_file = os.path.join(base_path_query, "news_dataset.db")

# Connect to the SQLite database
conn = sqlite3.connect(db_file)

# Read the dataset into a dataframe from the database
query = "SELECT * FROM news;"
df = pd.read_sql_query(query, conn)



# Feature 1: Ratio of stopwords to total text word count
df["stopword_to_length_ratio"] = df["text_stopword_count"] / (df["text_word_count"] + 1)

# Feature 2: Average word length in the title
df["title_avg_word_length"] = df["title"].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notnull(x) else 0)

# Feature 3: Word density = total words / sentences
df["word_density"] = df["text_word_count"] / (df["text_sentence_count"] + 1)

# Feature 4: Emotional word density in title
df["emotional_density"] = df["title_emotional_word_count"] / (df["title_word_count"] + 1)

# Feature 5: Interaction between question format and emotional intensity
df["question_emotion_interaction"] = df["is_question_title"] * df["title_emotional_word_count"]



# Save the updated dataframe back to the CSV file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "temp1.csv")
df.to_csv(data_file, index=False)