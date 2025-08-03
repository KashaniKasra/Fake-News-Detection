# 🧠 Fake News Detection - End-to-End Data Science Final Project

## 📚 Course: Data Science  
**Instructors:** Dr. Bahrak, Dr. Yaghoobzadeh  

---

## 🎯 Project Overview

This is a 3-phase real-world data science project focused on **Fake News Detection**, covering every stage of the pipeline from data storytelling and exploration to modeling and deployment.

We chose a high-complexity textual dataset containing real and fake news headlines/articles, explored insights using Power BI, processed and structured the data, then built, tracked, and deployed ML models.

---

## 🚦 Phase Breakdown

### 📌 Phase 1: Storytelling and Metadata

- Selected and documented a real-world fake news dataset
- Shared metadata: dataset name, source, sample entries
- Created a Power BI storytelling dashboard answering:
  - Which publishers post more fake news?
  - What’s the sentiment polarity of fake vs. real headlines?
  - What time periods show higher fake news activity?
- Delivered clear, insightful, and engaging visualizations

---

### 🧹 Phase 2: Preprocessing & Storage

- Structured dataset into a SQLite database
- Wrote modular Python scripts for:
  - Cleaning (stopwords, punctuation, lowercasing)
  - Tokenization and lemmatization
  - Storing intermediate results
- Ready-to-use CSVs and SQL dumps created
- Set up `pipeline-deployment.yaml` for reproducible runs

---

### 🤖 Phase 3: Modeling & Deployment

- Built training and prediction pipelines in Python
- Integrated `StandardScaler`, `TF-IDF`, and Logistic Regression
- Split data into train/val/test
- Tracked experiments using **MLflow**:
  - Logged models, metrics, and parameters
  - Compared experiments visually
- Achieved 0.89 accuracy, 0.91 AUC with final model
- Deployed predictions into the same database

---

## 🧠 Technologies Used

- **Python**: Data processing, modeling
- **Power BI**: Interactive visualizations
- **SQLite**: Lightweight storage and querying
- **MLflow**: Experiment tracking
- **Scikit-learn**: ML pipeline (TF-IDF, Logistic Regression)
- **YAML**: Deployment configs
- **Jupyter Notebook**: Phase 3 walkthrough

---

## 📁 Project Structure

```
📁 database/              → SQLite DB storing all data stages  
📁 notebook/              → Final notebooks 
📁 Training Pipeline/     → Scripts for training: preprocessing, modeling  
📁 Prediction Pipeline/   → Scripts for inference & result storage  
📁 scaler/                → Saved preprocessing objects  
📁 mlruns/                → MLflow artifacts  
📁 runID/                 → MLflow run tracking  
📄 pipeline-deployment.yaml → CI/CD config  
📄 requirements.txt       → Python dependencies  
📄 README.md              → Full documentation  
📄 P1/P2/P3.pdf           → Phase guidelines  
```

---

## ✅ Final Deliverables

- 📊 Power BI dashboard
- 📦 Structured database
- 🧪 Trained ML models + results
- 🧾 README documents for all phases
- 🎓 Fully reproducible end-to-end project

---

**Let the facts speak. No room for fakes.** 🔍📢