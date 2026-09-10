# Telco Customer Churn Prediction & Algorithmic Benchmark

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)

An end-to-end supervised machine learning benchmark on the **Telco Customer Churn** dataset, developed for the Graduate Machine Learning curriculum at Istanbul Technical University (ITU). 

The project evaluates binary classification of subscriber churn (`Churn: Yes/No`)[cite: 1, 2] through custom class-conditional data cleaning[cite: 1], unsupervised $K$-Means cluster feature augmentation, 5-fold hyperparameter grid searches, and a 10-fold stratified cross-validation benchmark comparing four distinct model architectures[cite: 2]: **Gradient Boosting**, **Random Forest**, **Support Vector Classifier (RBF Kernel)**, and a **5-Layer Deep Multi-Layer Perceptron (MLP)**[cite: 2].

---

## 📌 Problem & Dataset Architecture

The objective is binary classification of subscriber churn (`Churn = 0` vs. `Churn = 1`)[cite: 1, 2]:

* **Class Distribution:** The dataset exhibits class imbalance[cite: 2]—**73.4%** retained customers (`Churn = 0`) versus **26.6%** churned customers (`Churn = 1`)[cite: 2].
* **Feature Representation:** Demographic indicators (`gender`, `Partner`, `Dependents`)[cite: 1], subscribed customer services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`)[cite: 1], account structures (`Contract`, `PaperlessBilling`, `PaymentMethod`)[cite: 1], and quantitative billing metrics (`MonthlyCharges`, `TotalCharges`)[cite: 1].

---

## 🔧 Pipeline & Feature Engineering

### 1. Robust Data Cleaning (`cleaning_data.py`)
* **Whitespace Anomaly Imputation:** The `TotalCharges` field contained whitespace string values (`' '`) that evade standard missing value detection functions[cite: 1]. Rows containing whitespace were isolated, valid records cast to `float`, and missing values imputed using class-conditional mean imputation based on target labels[cite: 1].
* **Feature Encoding:** Binary features (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) were mapped directly to $\{0, 1\}$[cite: 1]. Non-binary categorical features were encoded into indicator columns via `pd.get_dummies`[cite: 1].

### 2. Unsupervised Feature Augmentation ($K$-Means Injection)
* Before cross-validation, an unsupervised $K$-Means clustering model ($k = 55$, `init='k-means++'`, `n_init=10`, `max_iter=10000000`, `tol=1e-04`, `random_state=42`) was fitted across the input feature space[cite: 2].
* Predicted cluster assignments were appended as an engineered feature (`X['Cluster']`)[cite: 2].

### 3. Preprocessing & Scaling
* **Stratified Split:** Data was partitioned into training ($80\%$) and testing ($20\%$) subsets with label stratification (`stratify=y`, `random_state=42`)[cite: 2].
* **Standardization:** A `StandardScaler` was fitted strictly on the training set and applied to both training and test feature sets[cite: 2].

---

## ⚙️ Model Tuning & Cross-Validation Framework

Hyperparameters were tuned via 5-fold cross-validation (`GridSearchCV`, $cv=5$)[cite: 2], followed by 10-fold stratified cross-validation (`StratifiedKFold`, $k=10$, `shuffle=True`, `random_state=42`) tracking Accuracy, F1-Score, and ROC-AUC across training and test splits[cite: 2]:

| Model Architecture | Hyperparameter Search Space (`GridSearchCV`, cv=5) | Optimal Configuration |
| :--- | :--- | :--- |
| **Random Forest**[cite: 2] | `max_depth`: [2, 5, 10, 15, 30]<br>`n_estimators`: [30, 50, 100, 250, 500][cite: 2] | `max_depth=10`, `n_estimators=500`[cite: 2] |
| **Gradient Boosting**[cite: 2] | `learning_rate`: [0.001, 0.01, 0.1, 0.5]<br>`max_depth`: [2, 3, 5, 7, 10]<br>`n_estimators`: [30, 50, 100, 260, 500][cite: 2] | `max_depth=2`, `random_state=42`[cite: 2] |
| **Support Vector Classifier**[cite: 2] | `C`: [0.1, 1, 10, 100, 1000]<br>`gamma`: [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]<br>`kernel`: ['rbf'][cite: 2] | `C=100`, `gamma=0.001`, `kernel='rbf'`[cite: 2] |
| **Neural Network (MLP)**[cite: 2] | Hidden Layers: `[(500,400,300,200,100), (400,400,400,400,400), (300,300,300,300,300), (200,200,200,200,200)]`<br>`activation`: ['logistic', 'tanh', 'relu']<br>`alpha`: [0.0001, 0.001, 0.005]<br>`early_stopping`: [True, False][cite: 2] | 5 layers $\times$ 300 nodes, `tanh`, `early_stopping=True`, `learning_rate_init=0.0001`[cite: 2] |

---

## 📊 Benchmark Results (10-Fold Stratified CV)

Mean evaluation metrics aggregated across all 10 folds[cite: 2]:

| Model | Train F1 | Test F1 | Train Acc | Test Acc | Train AUC | Test AUC | Generalization Gap (Test vs Train F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting**[cite: 2] | 0.6341[cite: 2] | **0.5820**[cite: 2] | 0.8252[cite: 2] | **0.8005**[cite: 2] | 0.7440[cite: 2] | **0.7123**[cite: 2] | **-0.0521**[cite: 2] |
| **Random Forest**[cite: 2] | 0.7391[cite: 2] | 0.5765[cite: 2] | 0.8724[cite: 2] | 0.7992[cite: 2] | 0.8114[cite: 2] | 0.7087[cite: 2] | -0.1626[cite: 2] |
| **Support Vector Machine**[cite: 2] | 0.7455[cite: 2] | 0.5429[cite: 2] | 0.8720[cite: 2] | 0.7697[cite: 2] | 0.8192[cite: 2] | 0.6884[cite: 2] | -0.2027[cite: 2] |
| **Neural Network (MLP)**[cite: 2] | 0.5195[cite: 2] | 0.4984[cite: 2] | 0.7938[cite: 2] | 0.7839[cite: 2] | 0.6762[cite: 2] | 0.6656[cite: 2] | -0.0211[cite: 2] |

---

## 💡 Key Diagnostic Findings

* **Top Out-of-Fold Performance:** **Gradient Boosting** attained the highest test performance across all three target metrics: **80.05% Accuracy**, **0.7123 ROC-AUC**, and **0.5820 F1-Score**[cite: 2].
* **Variance & Overfitting Dynamics:** Both **Random Forest** and **Support Vector Machine** exhibited substantial overfitting, with test F1 drops of approximately **16.3%** and **20.3%** relative to training F1[cite: 2]. Gradient Boosting maintained a narrow generalization gap of **5.2%**[cite: 2].
* **Tabular Performance on Deep Architectures:** The 5-layer feed-forward Multi-Layer Perceptron produced lower discriminatory capability relative to tree-based ensembles on this feature set, yielding a test ROC-AUC of **0.6656** and test F1 of **0.4984**[cite: 2].

---

## 📁 Repository Structure

```text
├── cleaning_data.py             # Data preprocessing, anomaly handling, and class-conditional mean imputation
├── telco_churn_benchmark.ipynb  # Complete benchmarking pipeline: K-Means, GridSearch, and 10-fold CV
└── README.md                    # Project overview and experimental benchmark results
