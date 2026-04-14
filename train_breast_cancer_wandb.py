import warnings
warnings.filterwarnings("ignore")

import wandb
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression


wandb.init(
    project="breast-cancer-classifier",
    config={
        "test_size": 0.2,
        "random_state": 42,
        "max_iter": 500,
        "model": "LogisticRegression",
        "dataset": "sklearn breast cancer",
    },
)

config = wandb.config

# Load dataset
bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
y = pd.Series(bc.target, name="target")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.test_size,
    random_state=config.random_state,
    stratify=y,
)

# Build pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(max_iter=config.max_iter, random_state=config.random_state),
    ),
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

wandb.log(
    {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bc.target_names)
disp.plot(ax=ax, colorbar=False)
plt.title("Breast Cancer Confusion Matrix")
plt.tight_layout()
cm_path = "confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
wandb.log({"confusion_matrix": wandb.Image(cm_path)})

# Prediction sample table
results_df = pd.DataFrame({"actual": y_test.values[:20], "predicted": y_pred[:20]})
wandb.log({"prediction_sample": wandb.Table(dataframe=results_df)})

print("Model training complete.")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

wandb.finish()
