import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("Loan Default Prediction/accepted_2007_to_2018Q4.csv", nrows=10000)

# Filter relevant loan statuses
data = data[data['loan_status'].isin(['Fully Paid', 'Charged Off'])]
data['loan_status'] = data['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# Keep selected columns
columns_to_keep = ['loan_amnt', 'term', 'annual_inc', 'dti', 'purpose', 'loan_status']
data = data[columns_to_keep]

# One-hot encode 'purpose'
data = pd.get_dummies(data, columns=['purpose'], drop_first=True)

# Clean 'term'
data['term'] = data['term'].str.replace(' months', '').astype(int)

# Define features and target
feature_cols = ['loan_amnt', 'term'] + [col for col in data.columns if col.startswith('purpose_')]
X = data[feature_cols]
y = data['loan_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression with class balancing
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Get probabilities for the positive class
y_proba = model.predict_proba(X_test)[:, 1]

# Apply final threshold
threshold = 0.45
y_pred_final = (y_proba >= threshold).astype(int)

# Evaluation
print("FINAL EVALUATION WITH THRESHOLD 0.45")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

thresholds = np.arange(0.1, 0.91, 0.05)
prec_0s, rec_0s, f1_0s = [], [], []
prec_1s, rec_1s, f1_1s = [], [], []

for threshold in thresholds:
    y_pred_custom = (y_proba >= threshold).astype(int)

    prec_0s.append(precision_score(y_test, y_pred_custom, pos_label=0, zero_division=0))
    rec_0s.append(recall_score(y_test, y_pred_custom, pos_label=0, zero_division=0))
    f1_0s.append(f1_score(y_test, y_pred_custom, pos_label=0, zero_division=0))

    prec_1s.append(precision_score(y_test, y_pred_custom, pos_label=1, zero_division=0))
    rec_1s.append(recall_score(y_test, y_pred_custom, pos_label=1, zero_division=0))
    f1_1s.append(f1_score(y_test, y_pred_custom, pos_label=1, zero_division=0))

# Plot
plt.figure(figsize=(12, 6))

plt.plot(thresholds, prec_1s, label='Precision (Class 1)', marker='o')
plt.plot(thresholds, rec_1s, label='Recall (Class 1)', marker='o')
plt.plot(thresholds, f1_1s, label='F1 Score (Class 1)', marker='o')

plt.plot(thresholds, prec_0s, label='Precision (Class 0)', marker='x', linestyle='--')
plt.plot(thresholds, rec_0s, label='Recall (Class 0)', marker='x', linestyle='--')
plt.plot(thresholds, f1_0s, label='F1 Score (Class 0)', marker='x', linestyle='--')

plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Performance Metrics vs. Threshold for Both Classes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
