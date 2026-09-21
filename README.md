## Regression Track + Classification Track (Part A)

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
git clone <your-repo-url>
cd <your-repo-folder>

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

### Regression — Top Models (R², dollar-scale)
| Rank | Model | R² | RMSE (USD) | MAE (USD) |
|---|---|---|---|---|
| 1 | Gradient Boosting Regressor | *fill in* | | |
| 2 | Random Forest Regressor | | | |

### Classification Part A — All 5 Algorithms
| Algorithm | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Decision Tree | 0.9811 | 0.9814 | 0.9811 | 0.9810 | 0.9955 |
| SVM (SVC) | 0.9764 | 0.9773 | 0.9764 | 0.9765 | 0.9989 |
| K-Nearest Neighbors | 0.9243 | 0.9230 | 0.9243 | 0.9229 | 0.9864 |
| Logistic Regression | 0.9078 | 0.9088 | 0.9078 | 0.9071 | 0.9921 |
| Naive Bayes (Gaussian) | 0.7447 | 0.7849 | 0.7447 | 0.7351 | 0.9497 |

**Best Part-A model:** Decision Tree (highest accuracy/F1), though SVM has the highest ROC-AUC — worth discussing this trade-off explicitly in the viva. Naive Bayes underperforms noticeably, likely due to its conditional-independence assumption not holding well across correlated lifestyle features (e.g. `FCVC`, `CH2O`, `FAF`).

*(Part B algorithms — Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP — and the full 10-algorithm consolidated table are completed in Review 2.)*

## 8. Academic Integrity Note

Code scaffolding for this project was developed with AI assistance (Claude, Anthropic). Analysis, interpretation, feature engineering decisions, and written commentary are the team's own original work, per the course's academic integrity guidelines.

## 9. Team

| Name | Track Ownership |
|---|---|
| *Member 1* | *e.g. Regression — preprocessing & linear-family models* |
| *Member 2* | *e.g. Regression — tree-based ensembles* |
| *Member 3* | *e.g. Classification Part A — EDA, all 5 algorithms* |
