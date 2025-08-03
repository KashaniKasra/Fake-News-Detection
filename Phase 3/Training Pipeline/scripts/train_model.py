import os
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_score, recall_score, confusion_matrix, roc_curve, log_loss, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
import mlflow
import mlflow.sklearn


# Create the directory structure for the mlflow uri file
base_uri_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mlruns"))
uri_path = f"file:///{base_uri_path.replace(os.sep, '/')}"

mlflow.set_tracking_uri(uri_path)

mlflow.set_experiment("News_Classification_Training")

with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Create the directory structure for the dataset file
    base_path_read = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
    read_news_file = os.path.join(base_path_read, "final_news.csv")

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




    # Drop label column from x values
    X = df.drop(columns=["label"])
    y = df["label"]

    # Split data into train(80%) and validation(10%) and test(10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Scale train and validation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define the MLP model
    models = {"MLP": (MLPClassifier(max_iter=300), {
            "hidden_layer_sizes": [(64,), (128, 64)],
            "activation": ['relu'],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001, 0.01]
        })
    }

    # Train our model with hyperparameter tuning
    best_model = {}
    for name, (model, param_grid) in models.items():
        print(f"{name} hyperparameter tuning...")

        grid = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=1)
        grid.fit(X_train_scaled, y_train)
        
        best_model[name] = grid.best_estimator_

    # Evaluate the best model on validation set
    for name, model in best_model.items():
        print(f"{name} best model evaluating...")

        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        print(classification_report(y_val, y_pred))

    # Test MLP model on test set
    model = best_model["MLP"]

    print(f"\nMLP model testing...")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred))




    mlflow.log_param("model_type", "MLP")
    mlflow.log_param("hidden_layer_sizes", model.hidden_layer_sizes)
    mlflow.log_param("activation", model.activation)

    mlflow.log_metric("train_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("train_precision", precision_score(y_test, y_pred))
    mlflow.log_metric("train_recall", recall_score(y_test, y_pred))
    mlflow.log_metric("train_f1", f1_score(y_test, y_pred))
    mlflow.log_metric("train_log_loss", log_loss(y_test, y_proba))
    mlflow.log_metric("train_AUC-ROC", roc_auc_score(y_test, y_proba))
    mlflow.log_metric("train_MCC", matthews_corrcoef(y_test, y_pred))
    mlflow.log_metric("train_Cohens_Kappa", cohen_kappa_score(y_test, y_pred))
    mlflow.log_metric("train_balanced_accuracy", balanced_accuracy_score(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("conf_matrix.png")
    plt.close()

    mlflow.log_artifact("conf_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    mlflow.log_artifact("roc_curve.png")




    # Create the directory structure for the scaler file
    base_scaler_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scaler"))
    scaler_path = os.path.join(base_scaler_path, "scaler_model.pkl")

    # Save the scaler
    joblib.dump(scaler, scaler_path)

    # Save the best MLP model
    mlflow.sklearn.log_model(best_model["MLP"], "model")

    # Create the directory structure for the runID
    base_runID_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runID"))
    runID_path = os.path.join(base_runID_path, "run_id.txt")

    # Save the run ID
    with open(runID_path, "w") as f:
        f.write(run_id)

    print("\nTraining completed and model saved to MLflow!")