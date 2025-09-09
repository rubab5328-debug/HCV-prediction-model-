# HCV-prediction-model-
# HCV Classification with Random Forest & SMOTE

This project applies **Machine Learning** techniques to classify individuals as **Healthy Donors** or **HCV Patients** (Hepatitis, Fibrosis, Cirrhosis) using a **Random Forest Classifier**.  
The main focus is to explore how **SMOTE (Synthetic Minority Oversampling Technique)** affects model performance on an imbalanced medical dataset.

---

## Dataset
- **Source:** Hepatitis C Virus (HCV) dataset  
- **Target Variable:** `Category`  
  - `0 = Healthy / Blood Donor`  
  - `1 = HCV Patient (Hepatitis / Fibrosis / Cirrhosis)`  
- **Features used:**
  - `ALT, AST, BIL, ALP, CHOL, PROT, ALB, Age`

---

## Workflow
1. **Data Preprocessing**
   - Cleaned column names
   - Mapped categories into binary (Healthy vs HCV)
   - Handled missing values with `SimpleImputer (mean strategy)`

2. **Train/Test Split**
   - 80% training, 20% testing
   - Stratified sampling to preserve class ratios

3. **Model 1: Random Forest (Without SMOTE)**
   - Trained directly on imbalanced dataset

4. **Model 2: Random Forest (With SMOTE)**
   - Balanced training set using **SMOTE**
   - Retrained Random Forest on balanced data

5. **Evaluation**
   - Classification report (precision, recall, f1-score, accuracy)
   - Confusion matrices (before/after SMOTE)
   - Feature importance
   - ROC curve comparison

---

## Results

- **Accuracy (both with & without SMOTE): ~96%**
- **Recall for HCV patients:** ~80%
- **Observation:**  
  SMOTE did **not significantly improve results** because:
  - Dataset not extremely imbalanced
  - Random Forest handles imbalance relatively well
  - Small test set (only ~15 HCV patients) → hard to see differences

---

## Key Insights
- Random Forest is strong for medical classification tasks, even on imbalanced data.  
- SMOTE is useful, but its impact depends on dataset size, imbalance severity, and model choice.  
- In healthcare applications, **recall is critical** (don’t miss sick patients!) → class weighting or threshold tuning may be more useful than oversampling alone.

---

## Visualizations
- Confusion matrices (before vs after SMOTE)
- Feature importance (top predictive lab values)
- ROC curve comparison

---

## How to Run

```bash
# Clone repository
git clone https://github.com/rubab5328-debug/hcv-classification.git
cd hcv-classification

# Install dependencies
pip install -r requirements.txt

# Run script
python hcv_rf_smote.py
