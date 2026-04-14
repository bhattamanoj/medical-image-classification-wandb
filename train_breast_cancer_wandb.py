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

import os
wandb.login(key=os.getenv("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"))

run = wandb.init(
    entity="bhattamanoj905",
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

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.test_size,
    random_state=config.random_state,
    stratify=y,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=config.max_iter, random_state=config.random_state)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

wandb.log({
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
})

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(ax=ax, colorbar=False)
plt.title("Breast Cancer Confusion Matrix")
plt.tight_layout()

cm_path = "confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)

wandb.log({"confusion_matrix": wandb.Image(cm_path)})

results_df = pd.DataFrame({
    "actual": y_test.values[:20],
    "predicted": y_pred[:20],
})
wandb.log({"prediction_sample": wandb.Table(dataframe=results_df)})

print("Model training complete.")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")

run.finish()