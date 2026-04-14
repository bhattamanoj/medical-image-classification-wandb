# Medical Classification with Weights & Biases

## Project Overview
This project shows how we can use Weights & Biases to track and explain a machine learning experiment in a simple and organized way. We used the Breast Cancer dataset from scikit-learn and trained a Logistic Regression model, then logged the main results to W&B.

## Problem Statement
When we test models without proper tracking, it becomes hard to compare runs, review metrics, and explain what happened during training. This project solves that by using W&B to keep the experiment clear and easy to follow.

## Solution
We trained a binary classifier and logged the main evaluation results to Weights & Biases. The project includes:
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- prediction sample table

## Tools and Technologies
- Python
- scikit-learn
- Weights & Biases
- pandas
- matplotlib

## Files
- `train_breast_cancer_wandb.py` – model training and W&B logging script
- `requirements.txt` – project dependencies

## How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Log in to W&B:
   `wandb login`
3. Run the project:
   `python train_breast_cancer_wandb.py`

## Key Results
This project gives us a clean example of experiment tracking in a real workflow. It helps show how monitoring tools can support reproducibility, comparison, and better reporting.

## Results

The model was trained and evaluated successfully.

- Accuracy: 0.9825  
- Precision: 0.9861  
- Recall: 0.9861  
- F1 Score: 0.9861  

## Visualization

![Confusion Matrix](assets/confusion_matrix.png)

## What We Learned
We learned how to connect a machine learning script to Weights & Biases, log useful metrics, and make experiment results easier to understand and present.

## Team Member
- Manoj Bhatta
