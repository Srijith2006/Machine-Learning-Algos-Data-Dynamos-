"""Regression (10 algorithms) and classification (Part A: 5, Part B: 5) engines.
Now fully generic: works with ANY uploaded CSV. The user picks target and feature columns;
pipelines auto-detect numeric vs categorical and build imputation/encoding/scaling dynamically."""
import ast
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler,
                                   label_binarize)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42
DATA = Path(__file__).resolve().parent.parent / "data"


# ============================== parameter spec helpers ==============================
# spec tuple: (key, label, kind, options-or-(min,max,step), default).  kinds: float,int,select,optint (0 -> None)
def build_params(spec, values):
    out = {}
    for key, _, kind, _, default in spec:
        v = values.get(key, default)
        if kind == "optint":
            v = None if int(v) == 0 else int(v)
        elif kind == "int":
            v = int(v)
        elif kind == "float":
            v = float(v)
        elif kind == "select" and isinstance(v, str) and v.startswith("("):
            v = ast.literal_eval(v)
        out[key] = v
    return out


def defaults(spec):
    return {k: d for k, _, _, _, d in spec}


# ================================== REGRESSION ======================================

class NumericPolyExpander(BaseEstimator, TransformerMixin):
    """PolynomialFeatures on the first n_numeric (numeric) columns; encoded categoricals pass through."""
    def __init__(self, degree=2, n_numeric=5):
        self.degree, self.n_numeric = degree, n_numeric

    def fit(self, X, y=None):
        self.poly_ = PolynomialFeatures(degree=self.degree, include_bias=False).fit(X[:, :self.n_numeric])
        return self

    def transform(self, X):
        return np.hstack([self.poly_.transform(X[:, :self.n_numeric]), X[:, self.n_numeric:]])


REG_SPECS = {
    "Linear Regression": dict(cls=LinearRegression, fixed={}, step="model",
        spec=[("fit_intercept", "fit_intercept", "select", [True, False], True)],
        grid={"model__fit_intercept": [True, False], "model__positive": [True, False]}),
    "Ridge Regression": dict(cls=Ridge, fixed={"random_state": SEED}, step="model",
        spec=[("alpha", "alpha (L2 strength)", "float", (0.0001, 1000.0, 0.1), 1.0)],
        grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]}),
    "Lasso Regression": dict(cls=Lasso, fixed={"random_state": SEED, "max_iter": 10000}, step="model",
        spec=[("alpha", "alpha (L1 strength)", "float", (0.00001, 10.0, 0.0005), 1.0)],
        grid={"model__alpha": [0.0001, 0.001, 0.01, 0.1]}),
    "ElasticNet Regression": dict(cls=ElasticNet, fixed={"random_state": SEED, "max_iter": 10000}, step="model",
        spec=[("alpha", "alpha", "float", (0.00001, 10.0, 0.0005), 1.0), ("l1_ratio", "l1_ratio", "float", (0.0, 1.0, 0.05), 0.5)],
        grid={"model__alpha": [0.0001, 0.001, 0.01], "model__l1_ratio": [0.2, 0.5, 0.8]}),
    "Polynomial Regression": dict(cls=LinearRegression, fixed={}, step="model", poly=True,
        spec=[("degree", "polynomial degree", "select", [1, 2, 3], 2)],
        grid={"poly__degree": [1, 2, 3]}),
    "Decision Tree Regressor": dict(cls=DecisionTreeRegressor, fixed={"random_state": SEED}, step="model",
        spec=[("max_depth", "max_depth (0 = unlimited)", "optint", (0, 30, 1), 0), ("min_samples_split", "min_samples_split", "int", (2, 50, 1), 2)],
        grid={"model__max_depth": [None, 4, 6, 8, 10, 15], "model__min_samples_split": [2, 5, 10]}),
    "Random Forest Regressor": dict(cls=RandomForestRegressor, fixed={"random_state": SEED, "n_jobs": -1}, step="model",
        spec=[("n_estimators", "n_estimators", "int", (10, 500, 10), 100), ("max_depth", "max_depth (0 = unlimited)", "optint", (0, 40, 1), 0)],
        grid={"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 10, 20]}),
    "Gradient Boosting Regressor": dict(cls=GradientBoostingRegressor, fixed={"random_state": SEED}, step="model",
        spec=[("learning_rate", "learning_rate", "float", (0.001, 1.0, 0.01), 0.1), ("n_estimators", "n_estimators", "int", (50, 500, 10), 100)],
        grid={"model__learning_rate": [0.01, 0.05, 0.1], "model__n_estimators": [100, 200]}),
    "Support Vector Regressor (SVR)": dict(cls=SVR, fixed={}, step="model",
        spec=[("C", "C", "float", (0.1, 100.0, 0.5), 1.0), ("kernel", "kernel", "select", ["rbf", "linear", "poly"], "rbf")],
        grid={"model__C": [1, 10, 50], "model__kernel": ["rbf", "linear"]}),
    "K-Nearest Neighbors Regressor": dict(cls=KNeighborsRegressor, fixed={}, step="model",
        spec=[("n_neighbors", "k (n_neighbors)", "int", (1, 40, 1), 5), ("weights", "weights", "select", ["uniform", "distance"], "uniform")],
        grid={"model__n_neighbors": [3, 5, 7, 9, 11, 15], "model__weights": ["uniform", "distance"]}),
}


def reg_prepare(raw, target_col, feature_cols, test_size=0.2, log_target=True):
    """Generic regression data preparation.
    Auto-detects numeric vs categorical features and builds column lists for the pipeline.
    """
    if target_col not in raw.columns:
        raise ValueError(f"Target column '{target_col}' not found in the CSV.")
    miss_feats = [c for c in feature_cols if c not in raw.columns]
    if miss_feats:
        raise ValueError(f"Feature columns not found: {miss_feats}")
    if not feature_cols:
        raise ValueError("Select at least one feature column.")

    d = raw.copy()
    # Clean target
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    n0 = len(d)
    d = d.dropna(subset=[target_col])
    d = d.drop_duplicates().copy()
    dups = n0 - len(d)

    if len(d) < 30:
        raise ValueError(f"Only {len(d)} usable rows after cleaning (need at least 30).")

    # Separate numeric and categorical features
    num_cols = []
    cat_cols = []
    for c in feature_cols:
        if c == target_col:
            continue
        col_data = pd.to_numeric(d[c], errors="coerce")
        if col_data.notna().sum() > len(d) * 0.5:  # >50% parseable as numeric
            num_cols.append(c)
            d[c] = col_data
        else:
            cat_cols.append(c)

    if not num_cols and not cat_cols:
        raise ValueError("No usable feature columns found.")

    # Target transform
    y_raw = d[target_col].values
    if log_target and (y_raw > 0).all():
        y = np.log1p(y_raw)
        target_transform = "log1p"
    else:
        y = y_raw.astype(float)
        target_transform = "none"

    X = d[num_cols + cat_cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=SEED)

    return dict(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        num_cols=num_cols, cat_cols=cat_cols,
        target_col=target_col, target_transform=target_transform,
        meta=dict(rows_raw=len(raw), duplicates=dups, rows=len(d)),
    )


def _reg_pre(num_cols, cat_cols):
    """Build a generic preprocessing ColumnTransformer."""
    transformers = []
    if num_cols:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02, sparse_output=False))])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers)


def reg_pipeline(name, params, data):
    s = REG_SPECS[name]
    num_cols, cat_cols = data["num_cols"], data["cat_cols"]
    steps = [("preprocessor", _reg_pre(num_cols, cat_cols))]
    if s.get("poly"):
        steps.append(("poly", NumericPolyExpander(degree=int(params.get("degree", 2)), n_numeric=len(num_cols))))
        steps.append(("model", s["cls"](**s["fixed"])))
    else:
        steps.append(("model", s["cls"](**s["fixed"], **params)))
    return Pipeline(steps)


def reg_metrics(y_true, y_pred, target_transform="none"):
    if target_transform == "log1p":
        yt, yp = np.expm1(y_true), np.expm1(np.clip(y_pred, -50, 50))
        return dict(R2=r2_score(yt, yp), RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
                    MAE=mean_absolute_error(yt, yp), R2_log=r2_score(y_true, y_pred))
    else:
        return dict(R2=r2_score(y_true, y_pred), RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    MAE=mean_absolute_error(y_true, y_pred), R2_log=r2_score(y_true, y_pred))


def reg_run(name, params, data, tune=False, cv_scores=False):
    """Fit one regression pipeline. tune=True runs GridSearchCV (5-fold, scoring r2)."""
    s = REG_SPECS[name]
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    tt = data.get("target_transform", "none")
    out = {"name": name}
    base = reg_pipeline(name, defaults_params(s), data).fit(Xtr, ytr)
    out["baseline"] = reg_metrics(yte, base.predict(Xte), tt)
    if tune:
        g = GridSearchCV(reg_pipeline(name, defaults_params(s), data), s["grid"], scoring="r2", cv=5, n_jobs=-1).fit(Xtr, ytr)
        pipe, out["best_params"], out["cv_best"] = g.best_estimator_, g.best_params_, g.best_score_
    else:
        pipe = reg_pipeline(name, params, data).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    out.update(pipe=pipe, metrics=reg_metrics(yte, pred, tt), pred_log=pred)
    if cv_scores:
        out["cv"] = cross_val_score(pipe, Xtr, ytr, scoring="r2", cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    return out


def defaults_params(s):
    return build_params(s["spec"], {})


def reg_importance(res, data):
    pipe, name = res["pipe"], res["name"]
    m = pipe.named_steps["model"]
    try:
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        names = None
    if hasattr(m, "feature_importances_") and names is not None and len(names) == len(m.feature_importances_):
        return pd.DataFrame({"feature": names, "importance": m.feature_importances_}).nlargest(12, "importance"), "Native feature importance"
    if hasattr(m, "coef_") and names is not None and np.ravel(m.coef_).shape[0] == len(names):
        c = np.ravel(m.coef_)
        return pd.DataFrame({"feature": names, "importance": c}).reindex(pd.Series(np.abs(c)).nlargest(12).index), "Coefficients (scaled units)"
    r = permutation_importance(pipe, data["X_test"], data["y_test"], n_repeats=5, random_state=SEED, scoring="r2")
    feat_names = list(data["num_cols"]) + list(data["cat_cols"])
    return pd.DataFrame({"feature": feat_names, "importance": r.importances_mean}).sort_values("importance", ascending=False), "Permutation importance (drop in R² when shuffled)"


# ================================ CLASSIFICATION ====================================

HL = ["(100,)", "(50,)", "(100, 50)", "(64, 32)"]
CLF_SPECS = {
    "Logistic Regression": dict(part="A", cls=LogisticRegression, fixed={"max_iter": 2000, "random_state": SEED},
        spec=[("C", "C (inverse regularisation)", "float", (0.001, 100.0, 0.1), 1.0)], grid={"C": [0.01, 0.1, 1, 10, 100]}),
    "K-Nearest Neighbors": dict(part="A", cls=KNeighborsClassifier, fixed={},
        spec=[("n_neighbors", "k (n_neighbors)", "int", (1, 41, 1), 5), ("metric", "distance metric", "select", ["euclidean", "manhattan", "chebyshev"], "euclidean")],
        grid={"n_neighbors": list(range(3, 22, 2)), "metric": ["euclidean", "manhattan"]}),
    "Naive Bayes (Gaussian)": dict(part="A", cls=GaussianNB, fixed={},
        spec=[("var_smoothing", "var_smoothing", "select", [1e-11, 1e-9, 1e-7, 1e-5, 1e-3], 1e-9)], grid={"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}),
    "Decision Tree Classifier": dict(part="A", cls=DecisionTreeClassifier, fixed={"random_state": SEED},
        spec=[("max_depth", "max_depth (0 = unlimited)", "optint", (0, 25, 1), 0), ("min_samples_leaf", "min_samples_leaf", "int", (1, 30, 1), 1),
              ("criterion", "criterion", "select", ["gini", "entropy"], "gini")],
        grid={"max_depth": list(range(2, 21, 2)), "min_samples_leaf": [1, 5, 10]}),
    "Support Vector Machine (SVC)": dict(part="A", cls=SVC, fixed={"probability": True, "random_state": SEED},
        spec=[("C", "C", "float", (0.1, 200.0, 0.5), 1.0), ("kernel", "kernel", "select", ["rbf", "linear", "poly"], "rbf")],
        grid={"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale"]}),
    "Random Forest Classifier": dict(part="B", cls=RandomForestClassifier, fixed={"random_state": SEED, "n_jobs": -1},
        spec=[("n_estimators", "n_estimators", "int", (10, 500, 10), 100), ("max_depth", "max_depth (0 = unlimited)", "optint", (0, 30, 1), 0)],
        grid={"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}),
    "AdaBoost Classifier": dict(part="B", cls=AdaBoostClassifier, fixed={"random_state": SEED},
        spec=[("n_estimators", "n_estimators", "int", (10, 500, 10), 50), ("learning_rate", "learning_rate", "float", (0.01, 2.0, 0.05), 1.0)],
        grid={"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]}),
    "Gradient Boosting Classifier": dict(part="B", cls=GradientBoostingClassifier, fixed={"random_state": SEED},
        spec=[("n_estimators", "n_estimators", "int", (50, 400, 10), 100), ("learning_rate", "learning_rate", "float", (0.01, 1.0, 0.01), 0.1),
              ("max_depth", "max_depth", "int", (1, 8, 1), 3)],
        grid={"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}),
    "Bagging Classifier": dict(part="B", cls=BaggingClassifier, fixed={"estimator": DecisionTreeClassifier(random_state=SEED), "random_state": SEED, "n_jobs": -1},
        spec=[("n_estimators", "n_estimators", "int", (5, 200, 5), 10), ("max_samples", "max_samples (fraction)", "float", (0.1, 1.0, 0.1), 1.0)],
        grid={"n_estimators": [10, 50, 100], "max_samples": [0.7, 1.0]}),
    "MLP Classifier (Neural Network)": dict(part="B", cls=MLPClassifier, fixed={"max_iter": 500, "random_state": SEED},
        spec=[("hidden_layer_sizes", "hidden layers", "select", HL, "(100,)"), ("activation", "activation", "select", ["relu", "tanh", "logistic"], "relu")],
        grid={"hidden_layer_sizes": [(50,), (100,), (100, 50)], "activation": ["relu", "tanh"]}),
}


# Part B (algorithms 6-10) is implemented but ENABLED. Set False to hide them from the UI.
INCLUDE_PART_B = True


def clf_active():
    """Classification algorithms currently exposed in the UI (read at call time, so the flag can be flipped)."""
    return {n: v for n, v in CLF_SPECS.items() if INCLUDE_PART_B or v["part"] == "A"}


def clf_prepare(raw, target_col, feature_cols, test_size=0.2):
    """Generic classification data preparation.
    Auto-detects numeric vs categorical features and builds a pipeline-ready dataset.
    """
    if target_col not in raw.columns:
        raise ValueError(f"Target column '{target_col}' not found in the CSV.")
    miss_feats = [c for c in feature_cols if c not in raw.columns]
    if miss_feats:
        raise ValueError(f"Feature columns not found: {miss_feats}")
    if not feature_cols:
        raise ValueError("Select at least one feature column.")

    d = raw.dropna(subset=[target_col]).drop_duplicates().copy()
    if len(d) < 30:
        raise ValueError(f"Only {len(d)} usable rows after cleaning (need at least 30).")

    # Encode target
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(d[target_col].astype(str)), name="target", index=d.index)

    # Separate numeric and categorical features
    num_cols = []
    cat_cols = []
    for c in feature_cols:
        if c == target_col:
            continue
        col_data = pd.to_numeric(d[c], errors="coerce")
        if col_data.notna().sum() > len(d) * 0.5:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    if not num_cols and not cat_cols:
        raise ValueError("No usable feature columns found.")

    # Build preprocessing pipeline
    transformers = []
    if num_cols:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02, sparse_output=False))])
        transformers.append(("cat", cat_pipe, cat_cols))
    preprocessor = ColumnTransformer(transformers)

    X = d[num_cols + cat_cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)

    # Fit preprocessor on training data only
    Xtr_processed = preprocessor.fit_transform(Xtr)
    Xte_processed = preprocessor.transform(Xte)

    try:
        col_names = list(preprocessor.get_feature_names_out())
    except Exception:
        col_names = [f"feature_{i}" for i in range(Xtr_processed.shape[1])]

    Xtr_df = pd.DataFrame(Xtr_processed, columns=col_names, index=Xtr.index)
    Xte_df = pd.DataFrame(Xte_processed, columns=col_names, index=Xte.index)

    return dict(
        X_train=Xtr_df, X_test=Xte_df, y_train=ytr, y_test=yte,
        preprocessor=preprocessor,
        num_cols=num_cols, cat_cols=cat_cols,
        classes=list(le.classes_), columns=col_names,
        label_encoder=le,
        meta=dict(rows=len(d), duplicates=int(raw.duplicated().sum())),
        raw=raw
    )


def clf_metrics(y_true, y_pred, proba, n_classes):
    result = dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
        Recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        F1=f1_score(y_true, y_pred, average="weighted"),
    )
    try:
        yb = label_binarize(y_true, classes=list(range(n_classes)))
        if yb.shape[1] == 1:
            # Binary case: label_binarize returns single column
            yb = np.hstack([1 - yb, yb])
        result["ROC_AUC"] = roc_auc_score(yb, proba, multi_class="ovr", average="weighted")
    except Exception:
        result["ROC_AUC"] = float("nan")
    return result


def clf_estimator(name, params):
    s = CLF_SPECS[name]
    return s["cls"](**s["fixed"], **params)


def clf_run(name, params, data, tune=False, cv_scores=False):
    s, n = CLF_SPECS[name], len(data["classes"])
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    out = {"name": name}
    base = clf_estimator(name, build_params(s["spec"], {})).fit(Xtr, ytr)
    out["baseline"] = clf_metrics(yte, base.predict(Xte), base.predict_proba(Xte), n)
    if tune:
        g = GridSearchCV(clf_estimator(name, build_params(s["spec"], {})), s["grid"], cv=5, scoring="f1_weighted", n_jobs=-1).fit(Xtr, ytr)
        model, out["best_params"], out["cv_best"] = g.best_estimator_, g.best_params_, g.best_score_
    else:
        model = clf_estimator(name, params).fit(Xtr, ytr)
    pred, proba = model.predict(Xte), model.predict_proba(Xte)
    out.update(model=model, metrics=clf_metrics(yte, pred, proba, n), pred=pred, proba=proba,
               cm=confusion_matrix(yte, pred, labels=list(range(n))),
               report=pd.DataFrame(classification_report(yte, pred, target_names=data["classes"], zero_division=0, output_dict=True)).T)
    if cv_scores:
        out["cv"] = cross_val_score(model, Xtr, ytr, scoring="f1_weighted", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    return out


def clf_importance(res, data):
    m = res["model"]
    cols = data["columns"]
    if hasattr(m, "feature_importances_"):
        return pd.DataFrame({"feature": cols, "importance": m.feature_importances_}).nlargest(12, "importance"), "Native feature importance"
    if hasattr(m, "coef_"):
        c = np.abs(m.coef_).mean(axis=0) if m.coef_.ndim > 1 else np.abs(m.coef_)
        return pd.DataFrame({"feature": cols, "importance": c}).nlargest(12, "importance"), "Mean |coefficient| across classes"
    r = permutation_importance(m, data["X_test"], data["y_test"], n_repeats=3, random_state=SEED, scoring="f1_weighted")
    return pd.DataFrame({"feature": cols, "importance": r.importances_mean}).nlargest(12, "importance"), "Permutation importance (drop in weighted F1)"


def clf_predict_row(res, data, row_values):
    """Predict a single row. row_values is a dict of raw feature values."""
    row_df = pd.DataFrame([row_values])
    X_processed = data["preprocessor"].transform(row_df[data["num_cols"] + data["cat_cols"]])
    X_df = pd.DataFrame(X_processed, columns=data["columns"])
    p = res["model"].predict_proba(X_df)[0]
    return int(np.argmax(p)), p
