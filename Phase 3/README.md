# 🧠 Fake News Detection - Final Project | Phase 3

## 📌 Course: Data Science  
**Instructors:** Dr. Bahrak, Dr. Yaghoobzadeh  

---

## 🎯 Phase 3 Goal: Final Model Development & End-to-End Deployment

This phase brings everything together: preprocessing, feature engineering, modeling, evaluation, automation, and deployment.

We designed and deployed a **robust pipeline** to detect fake news based on the data storytelling insights (Phase 1) and the processing pipeline (Phase 2).

---

## 🧩 Project Structure

```
📁 database/                → Final SQLite DB storing raw, cleaned, and prediction outputs  
📁 mlruns/                  → MLflow experiment tracking  
📁 notebook/                → Final Jupyter notebook (`Phase3_notebook.ipynb`)  
📁 scaler/                  → Saved StandardScaler objects (joblib/pkl)  
📁 Training Pipeline/       → train_model.py, feature_engineering.py, etc.  
📁 Prediction Pipeline/     → make_predictions.py, load_data.py, etc.  
📁 runID/                   → Tracks MLflow run references  
📄 P3.pdf                   → Instructions and phase requirements  
📄 pipeline-deployment.yaml → CI/CD setup with GitHub Actions  
📄 requirements.txt         → Python dependency list  
📄 README.md                → Project documentation  
```

---

## 🔧 Model Pipeline Overview

We used a full training + prediction pipeline based on `run_pipeline.py`:

### 🏗 Training Pipeline:
1. `load_data.py` → Load cleaned labeled data  
2. `preprocess.py` → Apply same transformations (scaling, encoding)  
3. `feature_engineering.py` → Generate TF-IDF & metadata features  
4. `train_model.py` → Logistic Regression + Hyperparameter tuning  
5. Save final model, scaler, and parameters

### 🔍 Prediction Pipeline:
1. Load new data from database  
2. Apply same preprocessing and feature extraction  
3. Load model and generate predictions  
4. Save predictions back to database

---

## 🧠 Final Model: Logistic Regression (TF-IDF + metadata)

We tested multiple classifiers (SVM, Naive Bayes, XGBoost, LR), and Logistic Regression performed best on our task considering:
- Accuracy: 0.89
- F1 Score: 0.87
- Precision/Recall: Balanced
- AUC: 0.91

---

## 📈 MLflow Tracking (Bonus +10%)

We used MLflow to:
- Track model parameters and metrics
- Store and retrieve saved models
- Assign run IDs for reproducibility
- Compare multiple versions easily

MLflow UI was served locally for experiment comparison.

---

## ✅ Phase Requirements Completed

- [x] Task definition + model selection
- [x] Data split into train/val/test
- [x] Proper preprocessing pipeline
- [x] Evaluation using metrics (Accuracy, F1, AUC)
- [x] Automation using `run_pipeline.py`
- [x] MLflow integration
- [x] Predictions saved into database

---

## 📁 Deliverable Format

- Filename: `DS_Project_P3_[student_number(s)].zip`
- Includes:
  - Final notebook
  - All pipeline scripts
  - Trained models & scalers
  - Database with predictions
  - README, requirements.txt, and P3.pdf

---