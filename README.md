# ML Capstone Lab — Clustering, Classification & Regression (23CSE301, 2026-27)

Interactive Streamlit app + executed Jupyter notebooks. Every number in the app comes from real scikit-learn/SciPy execution on the user's current settings; labels are never used to fit clustering models.

| Track | Algorithms in the app | Dataset |
|---|---|---|
| Regression | Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN | `enhanced_box_office_data_2000-2024.csv` (**not bundled, see below**) |
| Classification | Part A: Logistic, KNN, Gaussian NB, Decision Tree, SVC · Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP | `ObesityDataSet_raw_and_data_sinthetic.csv` (bundled) |
| Clustering | K-Means, Agglomerative (ward / complete / average / single) | World Happiness Report (bundled) |

## Repository layout
```
app/            app.py (navigation + clustering pages), supervised_pages.py (regression/classification UI),
                supervised.py (models, pipelines), utils.py (clustering helpers)
notebooks/      11_KMeans_Clustering.ipynb, 12_Agglomerative_Clustering.ipynb (executed, outputs visible)
data/           world_happiness_report.csv, ObesityDataSet_raw_and_data_sinthetic.csv
tests/          test_app.py (clustering UI), test_supervised_ui.py, make_box_office_fixture.py (SYNTHETIC test fixture only)
tools/          build_notebooks.py (regenerates the two notebooks)
RUBRIC_CHECKLIST.md
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app/app.py
```
**Regression needs your box-office CSV.** Your regression notebooks read `enhanced_box_office_data_2000-2024.csv`, which was not in the uploaded zip. Either copy it into `data/`, upload it from the sidebar on the Regression page, or set `BOX_OFFICE_CSV=/path/to/file.csv`.

## Run the notebooks
```bash
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 11_KMeans_Clustering.ipynb 12_Agglomerative_Clustering.ipynb
```

## Run the tests
```bash
python tests/test_app.py             # clustering UI (14 checks)
python tests/test_supervised_ui.py   # regression + classification UI (41 checks)
```

## Public deployment (bonus +1)
Not deployed by this build. To deploy: push this repo to GitHub, open https://share.streamlit.io, choose the repo, set the main file to `app/app.py`. Make sure the box-office CSV is committed to `data/` if you want the Regression page live.

## Modelling notes
- **Regression** follows your notebooks: leakage columns dropped, `log1p` target, metrics reported in USD (plus R² on the log scale), 80:20 random split with `random_state=42`, imputation/scaling/encoding inside the pipeline (fitted on training data only, also inside CV folds). One deliberate difference: the notebooks median-impute `Rating`/`Vote_Count` on the whole table before splitting; the app lets the pipeline do it on training data only (stricter, avoids leakage).
- **Classification** reproduces your notebook preprocessing exactly (verified: identical train/test matrices) and your saved Part A results.
- **Clustering** aggregates the happiness panel to one row per country; `Regional Indicator` is only used for post-hoc ARI validation.

## Known issue in the uploaded regression notebooks
Running `01_linear_regression.ipynb` from a clean kernel fails with `NameError: name 'ordinal_features' is not defined`. Fix: in the feature-set cell use
`categorical_features = ['Primary_Genre', 'Language_Grouped']` and add `ordinal_features = ['Decade']`. The saved outputs came from a stale kernel state, and the rubric requires notebooks to run top-to-bottom.

## AI assistance disclosure (required by the course guidelines)
Code scaffolding for the notebooks, app and tests was generated with Claude (Anthropic). The guidelines require analysis and interpretation to be your own: read the outputs, then write your own observations and conclusions before submission, and be ready to explain every part in the viva.
