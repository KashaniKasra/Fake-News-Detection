import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "temp5.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)
os.remove(data_file)



# Specify the features to scale
features_to_scale = [
    "title_capital_word_count",
    "title_emotional_word_count",
    "title_word_count",
    "text_word_count",
    "text_stopword_count",
    "text_stopword_ratio",
    "text_sentence_count",
    "text_lexical_diversity",
    "title_number_count",
    "text_number_count",
    "text_url_count",
    "stopword_to_length_ratio",
    "title_avg_word_length",
    "word_density",
    "emotional_density",
    "question_emotion_interaction"
]

# Scale and standardize the features
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Convert these 2 columns types to integer
df["publish_dayofweek"] = df["publish_dayofweek"].astype(int)
df["publish_month"] = df["publish_month"].astype(int)



# Save the updated dataframe back to the CSV file
data_file = os.path.join(base_path_data, "temp6.csv")
df.to_csv(data_file, index=False)