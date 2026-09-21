## Regression Track + Classification Track (Part A)

**Repository:** https://github.com/VIDHYADHARANRP7777/Machine-Learning-Algos-Data-Dynamos-

## 1. Problem Statements

**Regression Track:** Given metadata about a film (release year, audience rating, vote count, genre, language, production countries), predict its total **worldwide box-office gross revenue (`$Worldwide`)**.

**Classification Track (Part A):** Given a person's demographic, dietary, and lifestyle attributes, predict their **obesity level category (`NObeyesdad`)** — a 7-class classification problem.

## 2. Dataset Descriptions

### Regression — Enhanced Box Office Data (2000–2024)
- **Size:** 5,000 rows × 13 raw columns (top 200 releases/year, 2000–2024)
- **Target:** `$Worldwide` (USD), heavily right-skewed (skew ≈ 4.5) → modelled as `log1p($Worldwide)`
- **Leakage columns dropped:** `$Domestic`, `$Foreign`, `Domestic %`, `Foreign %`, `Rank` — all mathematically derived from `$Worldwide`
- **Missing values:** `Genres`, `Rating`, `Vote_Count`, `Original_Language`, `Production_Countries` (~3–4% each)

### Classification — Estimation of Obesity Levels (UCI)
- **Size:** 2,112 rows × 17 columns
- **Target:** `NObeyesdad` — 7 classes: `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`
- **Features:** `Age`, `Gender`, `Height`, `Weight`, `CALC` (alcohol consumption), `FAVC` (frequent high-calorie food), `FCVC` (vegetable consumption frequency), `NCP` (number of main meals), `SCC` (self-monitors calories), `SMOKE`, `CH2O` (daily water intake), `family_history_with_overweight`, `FAF` (physical activity frequency), `TUE` (tech/device usage hours), `CAEC` (snacking between meals), `MTRANS` (transportation mode)
- **Note:** `Height` and `Weight` are near-direct components of BMI, which strongly determines `NObeyesdad` — treated carefully to avoid the model trivially reconstructing the label rather than learning genuine patterns from lifestyle/dietary features
- **Class balance:** verified via `class_mapping.csv` and the target distribution plot in `00_eda_preprocessing.ipynb`

## 3. Repository Structure

/ (root)
├── README.md
├── requirements.txt
├── data/
│ ├── enhanced_box_office_data_2000-2024.csv
│ └── classification/
│ ├── ObesityDataSet_raw_and_data_sinthetic.csv
│ └── processed/
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ ├── y_test.csv
│ └── class_mapping.csv
├── notebooks/
│ ├── regression/
│ │ ├── 00_data_exploration.ipynb
│ │ ├── 01_linear_regression.ipynb
│ │ ├── 02_ridge_regression.ipynb
│ │ ├── 03_lasso_regression.ipynb
│ │ ├── 04_elasticnet_regression.ipynb
│ │ ├── 05_polynomial_regression.ipynb
│ │ ├── 06_decision_tree_regressor.ipynb
│ │ ├── 07_random_forest_regressor.ipynb
│ │ ├── 08_gradient_boosting_regressor.ipynb
│ │ ├── 09_support_vector_regressor.ipynb
│ │ ├── 10_knn_regressor.ipynb
│ │ └── regression_comparison.ipynb
│ └── classification_part_a/
│ ├── 00_eda_preprocessing.ipynb
│ ├── 01_logistic_regression.ipynb
│ ├── 02_knn.ipynb
│ ├── 03_naive_bayes.ipynb
│ ├── 04_decision_tree.ipynb
│ ├── 05_svm.ipynb
│ └── 06_comparison_table.ipynb
├── results/
│ └── classification_part_a/
├── models/
└── app/


## 4. Environment Setup

```bash
git clone https://github.com/VIDHYADHARANRP7777/Machine-Learning-Algos-Data-Dynamos-.git
cd Machine-Learning-Algos-Data-Dynamos-

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`:**

numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter


## 5. How to Run

```bash
jupyter notebook

# Regression track:
#   notebooks/regression/00_data_exploration.ipynb
#   notebooks/regression/01_linear_regression.ipynb ... 10_knn_regressor.ipynb
#   notebooks/regression/regression_comparison.ipynb

# Classification track (Part A):
#   notebooks/classification_part_a/00_eda_preprocessing.ipynb
#   notebooks/classification_part_a/01_logistic_regression.ipynb ... 05_svm.ipynb
#   notebooks/classification_part_a/06_comparison_table.ipynb
```

Notebooks expect data at `../../data/...` relative to their own folder — run from within each notebook's directory, or adjust `DATA_PATH` at the top.

## 6. Methodology Summary

### Regression
Cleaning (leakage removal, IQR outlier audit, duplicate check) → feature engineering (`Genre_Count`, `Primary_Genre`, `Country_Count`, `Language_Grouped`, `Decade`) → `ColumnTransformer`/`Pipeline` preprocessing fit on train only (`random_state=42` fixed across all notebooks) → 10 algorithms trained → `GridSearchCV` (5-fold) tuning → 5-fold CV R² → residual / predicted-vs-actual / feature-importance plots.

### Classification (Part A)
EDA + preprocessing (`00_eda_preprocessing.ipynb`) → 5 required Part-A algorithms (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM) each trained, tuned, and evaluated in their own notebook → consolidated in `06_comparison_table.ipynb` with Accuracy, weighted Precision/Recall/F1, confusion matrices, and One-vs-Rest ROC-AUC for all 5 models.

## 7. Results Summary

### Regression — All 10 Models (R², dollar-scale)
| Rank | Model | R² | RMSE (USD) | MAE (USD) |
|---|---|---|---|---|
| 1 | Gradient Boosting Regressor | 0.5899 | 138,322,848 | 57,875,904 |
| 2 | Support Vector Regressor | 0.5801 | 139,971,421 | 57,094,052 |
| 3 | Random Forest Regressor | 0.5608 | 143,140,211 | 58,510,894 |
| 4 | K-Nearest Neighbors Regressor | 0.5113 | 151,001,244 | 64,740,554 |
| 5 | Decision Tree Regressor | 0.4992 | 152,860,354 | 65,527,969 |
| 6 | Polynomial Regression (deg=2) | 0.4370 | 162,075,699 | 64,288,504 |
| 7 | Linear Regression | -6.5124 | 592,024,958 | 94,898,796 |
| 8 | Ridge Regression | -6.5143 | 592,100,792 | 94,862,647 |
| 9 | ElasticNet Regression | -6.6733 | 598,332,855 | 95,008,609 |
| 10 | Lasso Regression | -6.7614 | 601,759,223 | 95,127,225 |

**Top-2 hyperparameter tuning results:**

| Model | Best Params | R² (log-scale CV) | R² (tuned) | Improvement |
|---|---|---|---|---|
| Gradient Boosting Regressor | `{'learning_rate': 0.1, 'n_estimators': 200}` | 0.589905 | 0.645953 | +0.056048 |
| Support Vector Regressor | `{'C': 1, 'kernel': 'rbf'}` | 0.580071 | 0.580071 | 0.000000 |

**Best model:** Gradient Boosting Regressor (R² = 0.59 baseline, improving to 0.65 after tuning), driven primarily by `Vote_Count` and `Rating_numeric`.

> ⚠️ **Known issue to resolve before final submission:** Linear, Ridge, ElasticNet, and Lasso all show strongly **negative R²** (≈ -6.5 to -6.8), meaning they perform worse than a naive mean-prediction baseline. This is not expected behavior for a properly regularized linear model and should be root-caused (likely candidates: predictions not being inverse-transformed from log-scale correctly before scoring, a train/test preprocessing mismatch, or an extreme unclipped prediction blowing up the error) rather than reported as-is.

### Classification Part A — All 5 Algorithms
| Algorithm | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Decision Tree | 0.9811 | 0.9814 | 0.9811 | 0.9810 | 0.9955 |
| SVM (SVC) | 0.9764 | 0.9773 | 0.9764 | 0.9765 | 0.9989 |
| K-Nearest Neighbors | 0.9243 | 0.9230 | 0.9243 | 0.9229 | 0.9864 |
| Logistic Regression | 0.9078 | 0.9088 | 0.9078 | 0.9071 | 0.9921 |
| Naive Bayes (Gaussian) | 0.7447 | 0.7849 | 0.7447 | 0.7351 | 0.9497 |




| *Member 1* | *e.g. Regression — preprocessing & linear-family models* |
| *Member 2* | *e.g. Regression — tree-based ensembles* |
| *Member 3* | *e.g. Classification Part A — EDA, all 5 algorithms* |
