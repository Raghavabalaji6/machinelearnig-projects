Loan Default Prediction (ML Project)

A complete end-to-end Machine Learning project to predict whether a loan applicant will default (risky borrower) using XGBoost, threshold tuning, and proper handling of class imbalance.

🎯 Project Goal

Financial institutions want to avoid risky customers (loan defaulters).
The objective of this project is to:

Detect risky (default) customers with high RECALL

Reduce False Negatives — missing risky customers

Build a production-ready ML pipeline

Deploy a simple Streamlit prediction app

📂 Dataset Information

Size: ~255,000 rows
Target column: Default (1 = default, 0 = safe)

Features include:

Age, Income, LoanAmount, CreditScore

MonthsEmployed, NumCreditLines

InterestRate, LoanTerm, DTIRatio

Education, EmploymentType, MaritalStatus

HasMortgage, HasDependents, LoanPurpose, HasCoSigner

This dataset is imbalanced

Safe customers: ~88%

Risky customers: ~12%

🏗️ Project Pipeline
1️⃣ Data Cleaning

Dropped LoanID

Filled missing values

Encoded categorical columns using Label Encoding

Scaled numeric features with StandardScaler

2️⃣ Train/Test Split

Used:

train_test_split(..., stratify=y)


Ensures same class ratio in both train and test sets.

3️⃣ Handling Imbalance

Risky customers are only 12%, so we compute:

scale_pos_weight = neg / pos


and feed it into XGBoost → gives more weight to defaults.

4️⃣ Models Trained
Baseline

✔ Logistic Regression (class_weight="balanced")

Final Model

✔ XGBoost Classifier

n_estimators: 400

max_depth: 6

learning_rate: 0.05

tree_method: "hist"

scale_pos_weight: calculated from train data

5️⃣ Threshold Tuning (Most Important)

Instead of default 0.5, best results were at:

threshold = 0.30

📊 Final Results (XGBoost + threshold 0.30)
Confusion Matrix
[[18236, 26903],
 [  702,  5229]]

Key Metrics
Metric	Value	Why it matters
Recall (Default=1)	88.16%	Catches most risky customers
Precision	Low (expected)	Because recall is prioritized
Accuracy	~71%	Not a priority metric

High recall = fewer risky customers missed.

📈 Feature Importance

Top influential features:

CreditScore

DTIRatio

LoanAmount

Income

MonthsEmployed

🧩 Saved Artifacts

xgb_default_model.pkl

scaler.pkl

label_encoders.pkl

model_columns.pkl

threshold.pkl
