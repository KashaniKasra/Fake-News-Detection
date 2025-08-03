import os
import pandas as pd
import joblib
import mlflow


# Create the directory structure for the dataset file
base_path_read = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
read_news_file = os.path.join(base_path_read, "final_news_test.csv")

# Load the news dataset into a pandas dataframe
df = pd.read_csv(read_news_file)




def parse_space_separated_array(s):
    return [float(x) for x in s.strip("[]").split()]

df['text_embedding'] = df['text_embedding'].apply(parse_space_separated_array)
df['title_embedding'] = df['title_embedding'].apply(parse_space_separated_array)

text_embed_df = pd.DataFrame(df['text_embedding'].tolist(), columns=[f"text_emb_{i}" for i in range(len(df['text_embedding'].iloc[0]))])
title_embed_df = pd.DataFrame(df['title_embedding'].tolist(), columns=[f"title_emb_{i}" for i in range(len(df['title_embedding'].iloc[0]))])

df = df.drop(columns=['text_embedding', 'title_embedding'])
df = pd.concat([df, text_embed_df, title_embed_df], axis=1)





# Create the directory structure for the scaler file
base_scaler_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scaler"))
scaler_path = os.path.join(base_scaler_path, "scaler_model.pkl")

# Load the scaler file
scaler = joblib.load(scaler_path)

# Scale test set
X_test_scaled = scaler.transform(df)

# Create the directory structure for the runID
base_runID_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runID"))
runID_path = os.path.join(base_runID_path, "run_id.txt")

# Load the run ID from the file
with open(runID_path, "r") as f:
    run_id = f.read().strip()

# Set the MLflow tracking URI to the local directory
base_uri_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mlruns"))
uri_path = f"file:///{base_uri_path.replace(os.sep, '/')}"

mlflow.set_tracking_uri(uri_path)

# Load and test MLP model on test set from MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

print(f"\nMLP model testing...")

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]




# Create the directory structure for the predictions output file
base_output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))
output_path = os.path.join(base_output_path, "predictions.csv")

# Save the predictions output file
pd.DataFrame({"label": y_pred}).to_csv(output_path, index=False)

print("\n Testing completed and predictions saved to output!")