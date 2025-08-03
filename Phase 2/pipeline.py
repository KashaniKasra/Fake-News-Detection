import subprocess
import os

# Create the directory structure for the script files
base_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(base_dir, "scripts")

# Specifying the list of scripts to run in order
scripts = [
    "preprocess_featureExtract.py",
    "database_connection.py",
    "DB_queries.py",
    "advanced_feature_engineering.py",
    "text_vectorization.py",
    "encoding_categorical_variables.py",
    "handling_time_series.py",
    "handling_missing_data.py",
    "standardization.py",
    "removing_irrelevant_features.py",
    "DB_import_final_data.py"
]

# Running a series of Python scripts in sequence using subprocess
for script in scripts:
    script_path = os.path.join(scripts_dir, script)
    print(f"\n\n\tRunning {script} ...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    
    if result.stderr:
        print(f"\tError in {script}:\n{result.stderr}")