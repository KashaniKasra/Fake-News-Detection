import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Create the directory structure for the dataset file
base_path_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_data, exist_ok=True)
data_file = os.path.join(base_path_data, "temp1.csv")

# Read the CSV file into a dataframe
df = pd.read_csv(data_file, low_memory=False)
os.remove(data_file)



# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure title and text are strings
df["title"] = df["title"].astype(str)
df["text"] = df["text"].astype(str)

# Define embedding function
def get_bert_embedding(txt):
    inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Apply to title
tqdm.pandas()
df["title_embedding"] = df["title"].progress_apply(get_bert_embedding)

# Apply to text
df["text_embedding"] = df["text"].progress_apply(get_bert_embedding)



# Save the updated dataframe back to the CSV file
data_file = os.path.join(base_path_data, "temp2.csv")
df.to_csv(data_file, index=False)