from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer # Import SimpleImputer

# Load and preprocess
df = pd.read_csv("/content/HCV data.csv")
df.columns = df.columns.str.strip().str.replace(' ', '_')
df['Category'] = df['Category'].map({
    '0=Blood Donor': 0,
    '0s=suspect Blood Donor': 0,
    '1=Hepatitis': 1,
    '2=Fibrosis': 1,
    '3=Cirrhosis': 1
})

# Features & target
X = df[['ALT', 'AST', 'BIL', 'ALP', 'CHOL', 'PROT', 'ALB', 'Age']]
y = df['Category']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Impute missing values
imputer = SimpleImputer(strategy='mean') # Initialize SimpleImputer

# Fit on training data and transform both training and testing data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# --- MODEL WITHOUT SMOTE ---
# Train the model without SMOTE on the imputed training data
model_no_smote = RandomForestClassifier(random_state=42)
model_no_smote.fit(X_train_imputed, y_train)
y_pred_no_smote = model_no_smote.predict(X_test_imputed) # Predict on imputed test data
print("---- Without SMOTE ----")
print(classification_report(y_test, y_pred_no_smote, target_names=['Healthy', 'HCV Patient']))

# --- MODEL WITH SMOTE ---
smote = SMOTE(random_state=42)
# Apply SMOTE on the imputed training data
X_train_res, y_train_res = smote.fit_resample(X_train_imputed, y_train)
model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_train_res, y_train_res)
y_pred_smote = model_smote.predict(X_test_imputed) # Predict on imputed test data
print("---- With SMOTE ----")
print(classification_report(y_test, y_pred_smote, target_names=['Healthy', 'HCV Patient']))

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_no_smote), annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title("Without SMOTE")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_xticklabels(['Healthy', 'HCV'], va='center')
axes[0].set_yticklabels(['Healthy', 'HCV'], va='center')


sns.heatmap(confusion_matrix(y_test, y_pred_smote), annot=True, fmt='d', cmap="Greens", ax=axes[1])
axes[1].set_title("With SMOTE")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_xticklabels(['Healthy', 'HCV'], va='center')
axes[1].set_yticklabels(['Healthy', 'HCV'], va='center')
plt.tight_layout()
plt.show()


# Feature Importance
importances = model_smote.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.title("Feature Importance (With SMOTE)")
plt.show()

# ROC Curves
# Predict probabilities on imputed test data
y_prob_no_smote = model_no_smote.predict_proba(X_test_imputed)[:, 1]
y_prob_smote = model_smote.predict_proba(X_test_imputed)[:, 1]

fpr1, tpr1, _ = roc_curve(y_test, y_prob_no_smote)
fpr2, tpr2, _ = roc_curve(y_test, y_prob_smote)

plt.figure(figsize=(7, 5))
plt.plot(fpr1, tpr1, label=f"Without SMOTE (AUC = {auc(fpr1, tpr1):.2f})")
plt.plot(fpr2, tpr2, label=f"With SMOTE (AUC = {auc(fpr2, tpr2):.2f})", linestyle="--")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
