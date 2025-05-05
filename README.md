# 📰 Fake News Detection Pipeline

This project implements a complete **data processing and feature engineering pipeline** to analyze fake and true news articles. The pipeline takes raw CSVs (`Fake.csv`, `True.csv`) and processes them into a clean, engineered, and vectorized dataset ready for modeling.

---

## 🧭 Pipeline Steps

Each script represents a key phase in the pipeline. Run them sequentially using the `pipeline.py` file to fully prepare the data:

### 1. `preprocess_featureExtract.py`

* Loads and merges `Fake.csv` and `True.csv`.
* Cleans missing rows and formats the `date` column.
* Extracts features from title and text:

  * Capital word count
  * Question format detection
  * Emotional word count
  * Word/sentence/stopword counts and ratios
  * Lexical diversity, number counts, URL counts
  * Maps subjects into `general_category`
* Saves output to: `dataset/news.csv`

---

### 2. `database_connection.py`

* Reads `dataset/news.csv`
* Saves it as a SQLite table named `news` in `database/news_dataset.db`

---

### 3. `DB_queries.py`

* Runs 9 SQL queries over the `news` table:

  * Emotion averages, question ratios, hyperlinks, lexical diversity, etc.
* Stores query results as CSVs in the `queries/` folder.

---

### 4. `advanced_feature_engineering.py`

* Reads the table `news` from the SQLite database
* Adds new advanced features:

  * Stopword to text length ratio
  * Average title word length
  * Word density
  * Emotional density
  * Question-emotion interaction

---

### 5. `text_vectorization.py`

* Applies **BERT vectorization** on:

  * `title`
  * `text`
* Appends BERT embeded vectors as new columns to the DataFrame

---

### 6. `encoding_categorical_variables.py`

* Converts `label` to binary (0=fake, 1=true)
* One-hot encodes `general_category` and `subject`
* Converts boolean features to integers

---

### 7. `handling_time_series.py`

* Extracts temporal features from `date`:

  * Day of week
  * Month
  * Weekend indicator
  * Season

---

### 8. `handling_missing_data.py`

* Fills missing values in:

  * `date`: with default `2000-01-01`
  * `publish_dayofweek`, `publish_month`: random cyclic fill

---

### 9. `standardization.py`

* Standardizes numerical features using `StandardScaler`
* Converts `publish_dayofweek` and `publish_month` to integers

---

### 10. `removing_irrelevant_features.py`

* Removes:

  * Redundant text fields (`title`, `text`, `date`, etc.)
  * Columns with one unique value
  * Highly sparse features (>98% zeros)
  * Highly correlated features (Pearson > 90%)
* Final dataset is saved to: `dataset/final_news.csv`

---

## 🧩 About `pipeline.py`

The `pipeline.py` script is a master controller that automatically runs all processing stages in sequence using `subprocess.run()`. It improves automation and ensures that each step runs in the correct order without manual intervention.

## ⚙️ How to Run the Pipeline

1. Make sure `Fake.csv` and `True.csv` are in the `dataset/` folder.
2. Install dependencies in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the `pipeline.py` file.
4. Query results: `queries/*.csv`
5. Final cleaned dataset: `dataset/final_news.csv`

---

## 📓 Also Jupyter Notebook

For full explanations, code annotations, visualizations, and step-by-step output examples, please refer to the Jupyter Notebook located at `notebook/Phase2_notebook.ipynb`. This notebook walks through the pipeline stages interactively and includes insights, intermediate outputs, and additional commentary to support understanding and reproducibility. So you can only run this notebook once and see all results and ouputs just in there.