"""Regression and Classification pages. The sidebar dropdown selects ONE view: Dataset & EDA, a single algorithm, or Compare all."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import plot_tree

import supervised as S

PAL = px.colors.qualitative.Safe
ABOUT = {
    "Linear Regression": "Baseline ordinary least squares. Coefficients show each feature's linear effect on log gross.",
    "Ridge Regression": "Linear model with L2 regularisation (tune alpha). Shrinks coefficients, handles correlated features.",
    "Lasso Regression": "Linear model with L1 regularisation. Can shrink coefficients exactly to zero (feature sparsity).",
    "ElasticNet Regression": "Combines L1 and L2 penalties (tune alpha and l1_ratio).",
    "Polynomial Regression": "PolynomialFeatures on numeric inputs, then Linear Regression. Compare degrees 1-3.",
    "Decision Tree Regressor": "Splits the feature space into rectangles. Tune max_depth; inspect feature importance.",
    "Random Forest Regressor": "Ensemble of decorrelated trees (bagging). Tune n_estimators and max_depth.",
    "Gradient Boosting Regressor": "Sequentially fitted trees that correct previous errors. Tune learning_rate and n_estimators.",
    "Support Vector Regressor (SVR)": "Epsilon-insensitive regression with kernels. Features are scaled first; tune C and kernel.",
    "K-Nearest Neighbors Regressor": "Predicts the average of the k closest training points. Scaling matters; tune k.",
    "Logistic Regression": "Baseline linear classifier; coefficients give per-class log-odds.",
    "K-Nearest Neighbors": "Classifies by majority vote of the k nearest points. Tune k and the distance metric.",
    "Naive Bayes (Gaussian)": "Bayes rule with a conditional-independence assumption and Gaussian likelihoods.",
    "Decision Tree Classifier": "Learns if/else splits; the fitted tree can be visualised.",
    "Support Vector Machine (SVC)": "Maximum-margin classifier with kernels. Features are scaled; tune C and kernel.",
    "Random Forest Classifier": "Ensemble of trees with feature importance. Tune n_estimators.",
    "AdaBoost Classifier": "Boosting that re-weights misclassified samples. Tune n_estimators and learning_rate.",
    "Gradient Boosting Classifier": "Stage-wise boosted trees. Tune learning_rate and n_estimators.",
    "Bagging Classifier": "Bootstrap aggregation with a Decision Tree base estimator.",
    "MLP Classifier (Neural Network)": "Multi-layer perceptron. Tune hidden layer sizes and activation.",
}
EXTRA = {"Dataset & EDA": "📊 Dataset & EDA", "Compare all algorithms": "🏆 Compare all algorithms",
         "Evaluation Metrics": "📊 Evaluation Metrics",
         "Overview": "ℹ️ Overview", "Dataset & Preprocessing": "🧹 Dataset & Preprocessing",
         "K-Means Clustering": "1. K-Means Clustering", "Agglomerative Hierarchical Clustering": "2. Agglomerative Hierarchical Clustering",
         "Compare clustering algorithms": "🏆 Compare clustering algorithms"}


def view_label(task, view):
    """Label shown in the sidebar dropdown (numbered, grouped; the underlying value stays the plain algorithm name)."""
    if view in EXTRA:
        return EXTRA[view]
    names = list(S.REG_SPECS) if task == "Regression" else list(S.clf_active())
    i = names.index(view) + 1
    return f"{i}. {view}"


# --------------------------------------------------------------------- cached compute
@st.cache_data(show_spinner=False)
def c_reg_prepare(raw, test_size):
    return S.reg_prepare(raw, test_size)


@st.cache_data(show_spinner=False)
def c_clf_prepare(raw, test_size):
    return S.clf_prepare(raw, test_size)


@st.cache_data(show_spinner=False, max_entries=64)
def c_reg_run(name, params_json, dkey, tune, cv, _data):
    return S.reg_run(name, json.loads(params_json), _data, tune, cv)


@st.cache_data(show_spinner=False, max_entries=64)
def c_clf_run(name, params_json, dkey, tune, cv, _data):
    return S.clf_run(name, json.loads(params_json), _data, tune, cv)


@st.cache_data(show_spinner=False)
def c_knn_curve(Xtr, ytr, metric):
    cv = StratifiedKFold(5, shuffle=True, random_state=S.SEED)
    ks = list(range(1, 26, 2))
    return pd.DataFrame({"k": ks, "cv_f1": [cross_val_score(KNeighborsClassifier(n_neighbors=k, metric=metric), Xtr, ytr, cv=cv, scoring="f1_weighted").mean() for k in ks]})


# --------------------------------------------------------------------- shared widgets
def param_widgets(spec, key):
    vals, cols = {}, st.columns(max(1, min(len(spec), 3)))
    for i, (k, label, kind, opt, default) in enumerate(spec):
        c, wk = cols[i % len(cols)], f"{key}_{k}"
        if kind == "select":
            vals[k] = c.selectbox(label, opt, index=opt.index(default), key=wk)
        elif kind in ("int", "optint"):
            vals[k] = c.slider(label, int(opt[0]), int(opt[1]), int(default), step=int(opt[2]), key=wk)
        else:
            vals[k] = c.number_input(label, float(opt[0]), float(opt[1]), float(default), step=float(opt[2]), format="%.5g", key=wk)
    return vals


def grid_text(grid):
    return {k.split("__")[-1]: [str(v) for v in vs] for k, vs in grid.items()}


def fmt_metrics(d, kind):
    if kind == "reg":
        return {"R² (USD)": d["R2"], "RMSE (USD)": d["RMSE"], "MAE (USD)": d["MAE"], "R² (log scale)": d["R2_log"]}
    return {"Accuracy": d["Accuracy"], "Precision (wtd)": d["Precision"], "Recall (wtd)": d["Recall"], "F1 (wtd)": d["F1"], "ROC-AUC (OvR)": d["ROC_AUC"]}


def explain_reg_metrics():
    with st.expander("📊 Evaluation Metrics – Regression"):
        st.markdown("### Regression Evaluation Metrics")
        st.markdown("""
| Metric | Range | Ideal | Description |
|--------|-------|-------|-------------|
| **R² (Coefficient of Determination)** | −∞ to 1 | Higher (→ 1) | Proportion of variance explained by the model. 1 = perfect; 0 = as good as the mean; negative = worse than mean. |
| **RMSE (Root Mean Squared Error)** | ≥ 0 | Lower (→ 0) | Square root of the average of squared differences between predicted and actual values. Sensitive to large errors. |
| **MAE (Mean Absolute Error)** | ≥ 0 | Lower (→ 0) | Average of the absolute differences between predictions and actual values. More robust to outliers than RMSE. |
| **R² (log scale)** | −∞ to 1 | Higher (→ 1) | R² computed on log-transformed target, showing fit on the scale the models were trained on. |
""")


def explain_clf_metrics():
    with st.expander("📊 Evaluation Metrics – Classification"):
        st.markdown("### Classification Evaluation Metrics")
        st.markdown("""
| Metric | Range | Ideal | Description |
|--------|-------|-------|-------------|
| **Accuracy** | 0 to 1 | Higher (→ 1) | Fraction of correctly classified samples out of all samples. Can be misleading with imbalanced classes. |
| **Precision (weighted)** | 0 to 1 | Higher (→ 1) | Of all predicted positives, how many are truly positive. Weighted average across classes. |
| **Recall (weighted)** | 0 to 1 | Higher (→ 1) | Of all actual positives, how many were correctly identified. Weighted average across classes. |
| **F1 Score (weighted)** | 0 to 1 | Higher (→ 1) | Harmonic mean of Precision and Recall. Balances both concerns. Weighted average across classes. |
| **ROC-AUC (One-vs-Rest)** | 0 to 1 | Higher (→ 1) | Area under the ROC curve (True Positive Rate vs False Positive Rate). 0.5 = random; 1 = perfect separation. |
| **Confusion Matrix** | — | Diagonal | Grid showing predicted vs actual class counts. Strong diagonal = good classifier. |
""")


def cards(d, kind):
    f = fmt_metrics(d, kind)
    for c, (k, v) in zip(st.columns(len(f)), f.items()):
        c.metric(k, f"{v:,.0f}" if "USD" in k and "R²" not in k else f"{v:.4f}")


def importance_fig(df, title):
    f = px.bar(df.iloc[::-1], x="importance", y="feature", orientation="h", title=title, color_discrete_sequence=[PAL[0]])
    return f.update_layout(height=420, margin=dict(t=50, b=10), xaxis_title="Importance", yaxis_title="Feature")


def show_tuning(res, kind):
    if "best_params" not in res:
        return
    st.markdown("**GridSearchCV result** (5-fold)")
    st.write("Best parameters:", {k.split("__")[-1]: v for k, v in res["best_params"].items()})
    b, t = fmt_metrics(res["baseline"], kind), fmt_metrics(res["metrics"], kind)
    tbl = pd.DataFrame({"Untuned (sklearn defaults)": b, "Tuned": t}); tbl["Change"] = tbl["Tuned"] - tbl["Untuned (sklearn defaults)"]
    st.dataframe(tbl.style.format("{:,.4f}"), width="stretch")


def show_cv(res, label):
    if "cv" in res:
        st.write(f"**5-fold CV {label}:** " + ", ".join(f"{x:.4f}" for x in res["cv"]) + f"  →  mean **{res['cv'].mean():.4f}** ± {res['cv'].std():.4f}")


def ranked_table(rows, sort_key):
    df = pd.DataFrame(rows).sort_values(sort_key, ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def get_upload_or_default(label, key, hint):
    up = st.sidebar.file_uploader(label, type="csv", key=key)
    if up is not None:
        return pd.read_csv(up), f"uploaded file `{up.name}`"
    st.info(f"⬅️ **Upload a CSV file** in the sidebar to begin. {hint}")
    return None, None


def train_controls(prefix, name, spec_dict):
    """Hyperparameter mode + widgets + CV option + train button. Returns (clicked, params_json, tune, want_cv)."""
    st.caption(ABOUT.get(name, ""))
    mode = st.radio("Hyperparameters", ["Manual", "Tune with GridSearchCV (5-fold)"], horizontal=True, key=f"{prefix}_mode_{name}")
    tune = mode.startswith("Tune")
    if tune:
        st.info("Grid searched: " + json.dumps(grid_text(spec_dict["grid"])))
        params = {}
    else:
        params = param_widgets(spec_dict["spec"], f"{prefix}_{name}")
    want_cv = st.checkbox("Also compute 5-fold cross-validation", key=f"{prefix}_cv_{name}")
    clicked = st.button("▶ Train & evaluate", type="primary", key=f"{prefix}_go")
    return clicked, json.dumps(S.build_params(spec_dict["spec"], params), default=str), tune, want_cv


# --------------------------------------------------------------------- entry points
def reg_views():
    return ["Dataset & EDA", "Evaluation Metrics", *S.REG_SPECS, "Compare all algorithms"]


def clf_views():
    return ["Dataset & EDA", "Evaluation Metrics", *S.clf_active(), "Compare all algorithms"]


def render(task, view, hero):
    if view == "Evaluation Metrics":
        if task == "Regression":
            hero("Regression · 📊 Evaluation Metrics", "Understanding the metrics used to evaluate regression models.")
            explain_reg_metrics()
        else:
            hero("Classification · 📊 Evaluation Metrics", "Understanding the metrics used to evaluate classification models.")
            explain_clf_metrics()
        return
    (_regression if task == "Regression" else _classification)(view, hero)


# ===================================================================== REGRESSION
def _regression(view, hero):
    n = len(S.REG_SPECS)
    hero(f"Regression · {view_label('Regression', view)}", f"{n} algorithms · box-office worldwide gross (log target, metrics in USD) · trained live on your settings")
    raw, src = get_upload_or_default("Upload Box-office CSV", "reg_up",
        "Please upload the box-office CSV (e.g. enhanced_box_office_data_2000-2024.csv) in the sidebar to begin.")
    if raw is None:
        return
    test_size = st.sidebar.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05, key="reg_ts")
    try:
        data = c_reg_prepare(raw, test_size)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not prepare the data: {e}")
        return
    dkey = f"{pd.util.hash_pandas_object(raw, index=False).sum()}_{test_size}"
    m = data["meta"]
    a, b, c, d = st.columns(4)
    a.metric("Rows (after cleaning)", f"{m['rows']:,}"); b.metric("Duplicates removed", m["duplicates"])
    c.metric("Train rows", f"{len(data['X_train']):,}"); d.metric("Test rows", f"{len(data['X_test']):,}")
    st.caption(f"Source: {src}. Same random split (random_state=42) for every algorithm, so comparisons are fair.")
    if view == "Dataset & EDA":
        _reg_eda(raw, data)
    elif view == "Compare all algorithms":
        _reg_compare(data, dkey)
    else:
        _reg_model(view, data, dkey)


def _reg_eda(raw, data):
    st.markdown("**Preprocessing:** leakage columns (`$Domestic`, `$Foreign`, their %, `Rank`, `Release Group`) dropped; `Rating` parsed from `x.xxx/10`; "
                "engineered `Genre_Count`, `Primary_Genre`, `Country_Count`, `Language_Grouped` (top-8 languages + Other), `Decade` (ordinal). "
                "Imputation, scaling and one-hot encoding are inside the model pipeline, so they are fitted on training data only (also inside CV folds).")
    st.dataframe(raw.head(15), width="stretch")
    y = np.expm1(pd.concat([data["y_train"], data["y_test"]]))
    f = go.Figure()
    f.add_trace(go.Histogram(x=y, name="$Worldwide (raw)", nbinsx=50)); f.add_trace(go.Histogram(x=np.log1p(y), name="log1p($Worldwide)", nbinsx=50, xaxis="x2", opacity=.7))
    f.update_layout(barmode="overlay", title="Target distribution: raw vs log1p", xaxis=dict(title="USD"), xaxis2=dict(title="log1p(USD)", overlaying="x", side="top"), yaxis_title="Count", height=380)
    st.plotly_chart(f, width="stretch")
    num = pd.concat([data["X_train"][S.REG_NUM], data["y_train"].rename("log1p($Worldwide)")], axis=1)
    st.plotly_chart(px.imshow(num.corr().round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True, title="Correlation heatmap (training set)").update_layout(height=450), width="stretch")
    feat = st.selectbox("Feature vs target", S.REG_NUM, key="reg_eda_feat")
    st.plotly_chart(px.scatter(x=data["X_train"][feat], y=data["y_train"], opacity=.4, labels={"x": feat, "y": "log1p($Worldwide)"}, title=f"{feat} vs log1p($Worldwide)", color_discrete_sequence=[PAL[1]]).update_layout(height=380), width="stretch")


def _reg_model(name, data, dkey):
    spec = S.REG_SPECS[name]
    explain_reg_metrics()
    clicked, pjson, tune, want_cv = train_controls("reg", name, spec)
    if clicked:
        try:
            with st.spinner(f"Training {name}…" + (" (grid search can take a while)" if tune else "")):
                st.session_state.setdefault("reg_models", {})[name] = c_reg_run(name, pjson, dkey, tune, want_cv, data)
        except Exception as e:  # noqa: BLE001
            st.error(f"Training failed: {e}")
    res = st.session_state.get("reg_models", {}).get(name)
    if res is None:
        st.info("Set the hyperparameters and press **Train & evaluate**.")
        return
    st.subheader("Results")
    cards(res["metrics"], "reg")
    st.caption("R² in USD is dominated by a few blockbusters and can be very negative if a model over-predicts exponentially; R² (log scale) shows fit on the scale the models were trained on.")
    show_tuning(res, "reg"); show_cv(res, "R² (log scale)")
    yt = np.expm1(data["y_test"].to_numpy()); yp = np.expm1(np.clip(res["pred_log"], -50, 50)); resid = yt - yp
    logax = st.checkbox("Log axes for predicted-vs-actual", True, key="reg_logax")
    a, b = st.columns(2)
    f = px.scatter(x=yt, y=yp, opacity=.45, labels={"x": "Actual $Worldwide (USD)", "y": "Predicted (USD)"}, title="Predicted vs actual", log_x=logax, log_y=logax, color_discrete_sequence=[PAL[1]])
    lo, hi = max(float(min(yt.min(), 1)), 1), float(yt.max())
    f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line=dict(dash="dash", color="black")))
    a.plotly_chart(f.update_layout(height=400), width="stretch")
    f = px.scatter(x=yp, y=resid, opacity=.45, labels={"x": "Predicted (USD)", "y": "Residual (actual − predicted)"}, title="Residual plot", color_discrete_sequence=[PAL[0]])
    f.add_hline(y=0, line_dash="dash", line_color="black")
    b.plotly_chart(f.update_layout(height=400), width="stretch")
    with st.spinner("Computing feature importance…"):
        imp, title = S.reg_importance(res, data)
    st.plotly_chart(importance_fig(imp, title), width="stretch")
    st.subheader("🔮 Predict with this model")
    X = data["X_train"]
    with st.form("reg_pred"):
        c1, c2, c3 = st.columns(3)
        year = c1.number_input("Release year", 1990, 2035, 2018); rating = c2.slider("Rating (0-10)", 0.0, 10.0, 6.5, 0.1)
        votes = c3.number_input("Vote count", 0, 5_000_000, int(X["Vote_Count"].median()), step=100)
        c1, c2, c3, c4 = st.columns(4)
        genre = c1.selectbox("Primary genre", data["genres"]); lang = c2.selectbox("Language", data["languages"])
        ng = c3.number_input("Number of genres", 1, 8, 2); nc = c4.number_input("Number of production countries", 1, 8, 1)
        ok = st.form_submit_button("Predict worldwide gross")
    if ok:
        row = pd.DataFrame([{"Primary_Genre": genre, "Language_Grouped": lang, "Decade": f"{int(year)//10*10}s", "Year": year, "Rating_numeric": rating,
                             "Vote_Count": votes, "Genre_Count": ng, "Country_Count": nc}])
        try:
            pred = float(np.expm1(np.clip(res["pipe"].predict(row)[0], -50, 50)))
            st.success(f"Predicted worldwide gross: **${pred:,.0f}** ({name})")
            if f"{int(year)//10*10}s" not in data["decade_order"]:
                st.warning("This decade was not in the training data, so it is encoded as an unknown value.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")


def _reg_compare(data, dkey):
    n = len(S.REG_SPECS)
    explain_reg_metrics()
    st.markdown(f"Trains **all {n} algorithms** on the same split and ranks them by R² (USD). 5-fold CV is added for the two best.")
    cmode = st.radio("Mode", ["Quick (default hyperparameters)", "Tuned (GridSearchCV for every model — slower)"], horizontal=True, key="reg_cmode")
    if st.button(f"▶ Run all {n} algorithms", type="primary", key="reg_all"):
        tune_all, rows, bar = cmode.startswith("Tuned"), [], st.progress(0.0)
        try:
            for i, (nm, sp) in enumerate(S.REG_SPECS.items()):
                bar.progress(i / n, text=f"Training {nm}…")
                r = c_reg_run(nm, json.dumps(S.build_params(sp["spec"], {}), default=str), dkey, tune_all, False, data)
                st.session_state.setdefault("reg_models", {})[nm] = r
                rows.append({"Model": nm, **{k: r["metrics"][k] for k in ("R2", "RMSE", "MAE", "R2_log")}})
            bar.progress(1.0, text="Cross-validating the two best models…")
            tbl = ranked_table(rows, "R2"); cvs = {}
            for nm in tbl["Model"].head(2):
                cvs[nm] = c_reg_run(nm, json.dumps(S.build_params(S.REG_SPECS[nm]["spec"], {}), default=str), dkey, tune_all, True, data)["cv"]
            tbl["CV R² (5-fold, log)"] = tbl["Model"].map(lambda x: float(cvs[x].mean()) if x in cvs else np.nan)
            st.session_state["reg_table"] = tbl; bar.empty()
        except Exception as e:  # noqa: BLE001
            bar.empty(); st.error(f"Comparison failed: {e}")
    tbl = st.session_state.get("reg_table")
    if tbl is None:
        return
    st.dataframe(tbl.style.format({"R2": "{:.4f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2_log": "{:.4f}", "CV R² (5-fold, log)": "{:.4f}"}, na_rep="—"), width="stretch", hide_index=True)
    a, b = st.columns(2)
    for col, key, title in [(a, "R2", "R² (USD) by model"), (b, "R2_log", "R² (log scale) by model")]:
        col.plotly_chart(px.bar(tbl, x=key, y="Model", orientation="h", title=title, color="Model", color_discrete_sequence=PAL).update_layout(showlegend=False, height=420, yaxis=dict(autorange="reversed")), width="stretch")
    st.download_button("⬇ Download comparison (CSV)", tbl.to_csv(index=False).encode(), "regression_comparison.csv", "text/csv")


# ===================================================================== CLASSIFICATION
def _classification(view, hero):
    active = S.clf_active(); n = len(active)
    parts = "Part A" if n == 5 else "Part A + Part B"
    hero(f"Classification · {view_label('Classification', view)}", f"{n} algorithms ({parts}) · obesity level · weighted metrics, confusion matrix, ROC-AUC (OvR)")
    raw, src = get_upload_or_default("Upload Obesity CSV", "clf_up",
        "Please upload the obesity classification CSV (e.g. ObesityDataSet_raw_and_data_sinthetic.csv) in the sidebar to begin.")
    if raw is None:
        return
    test_size = st.sidebar.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05, key="clf_ts")
    try:
        data = c_clf_prepare(raw, test_size)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not prepare the data: {e}")
        return
    dkey = f"{pd.util.hash_pandas_object(raw, index=False).sum()}_{test_size}"
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{data['meta']['rows']:,}"); b.metric("Classes", len(data["classes"])); c.metric("Train rows", len(data["X_train"])); d.metric("Test rows", len(data["X_test"]))
    st.caption(f"Source: {src}. Stratified split, random_state=42; scaler fitted on the training set only (no leakage).")
    if view == "Dataset & EDA":
        _clf_eda(raw)
    elif view == "Compare all algorithms":
        _clf_compare(data, dkey)
    else:
        _clf_model(view, data, dkey)


def _clf_eda(raw):
    st.markdown("**Preprocessing:** engineered `BMI = Weight / Height²`; `CAEC`, `CALC` ordinal-encoded (no < Sometimes < Frequently < Always); binary columns mapped to 0/1; "
                "`MTRANS` one-hot encoded; numeric columns standardised (scaler fitted on the training set).")
    st.dataframe(raw.head(15), width="stretch")
    vc = raw["NObeyesdad"].value_counts().reset_index(); vc.columns = ["Class", "Count"]
    st.plotly_chart(px.bar(vc, x="Class", y="Count", color="Class", color_discrete_sequence=PAL, title="Target class distribution").update_layout(showlegend=False, height=380), width="stretch")
    enc = S.clf_encode(raw.drop(columns=["NObeyesdad"])); num = S.CLF_SCALE
    st.plotly_chart(px.imshow(enc[num].corr().round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True, title="Correlation heatmap: numeric features").update_layout(height=520), width="stretch")
    feat = st.selectbox("Feature vs class", num, key="clf_eda_feat")
    st.plotly_chart(px.box(raw.assign(BMI=enc["BMI"]), x="NObeyesdad", y=feat, color="NObeyesdad", color_discrete_sequence=PAL, title=f"{feat} by obesity level").update_layout(showlegend=False, height=420, xaxis_title="Class"), width="stretch")


def _clf_model(name, data, dkey):
    spec, classes = S.CLF_SPECS[name], data["classes"]
    explain_clf_metrics()
    if spec["part"] == "B":
        st.caption("Part B grids are defaults chosen by this app (not from your Part A notebooks).")
    clicked, pjson, tune, want_cv = train_controls("clf", name, spec)
    if tune and name in ("Logistic Regression", "Naive Bayes (Gaussian)"):
        st.caption("Note: your Part A notebooks used default settings for this model; this grid is an extra.")
    if clicked:
        try:
            with st.spinner(f"Training {name}…" + (" (grid search can take a while)" if tune else "")):
                st.session_state.setdefault("clf_models", {})[name] = c_clf_run(name, pjson, dkey, tune, want_cv, data)
        except Exception as e:  # noqa: BLE001
            st.error(f"Training failed: {e}")
    res = st.session_state.get("clf_models", {}).get(name)
    if res is None:
        st.info("Set the hyperparameters and press **Train & evaluate**.")
        return
    st.subheader("Results")
    cards(res["metrics"], "clf"); show_tuning(res, "clf"); show_cv(res, "weighted F1")
    a, b = st.columns([3, 2])
    cm = pd.DataFrame(res["cm"], index=classes, columns=classes)
    a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion matrix (test set)").update_layout(height=520), width="stretch")
    b.markdown("**Per-class report**"); b.dataframe(res["report"].round(3), width="stretch", height=470)
    imp, title = S.clf_importance(res, data)
    st.plotly_chart(importance_fig(imp, title), width="stretch")
    if name == "K-Nearest Neighbors":
        curve = c_knn_curve(data["X_train"], data["y_train"], getattr(res["model"], "metric", "euclidean"))
        st.plotly_chart(px.line(curve, x="k", y="cv_f1", markers=True, title="KNN: 5-fold CV weighted F1 vs k (training set)", labels={"cv_f1": "CV weighted F1"}).update_layout(height=360), width="stretch")
        st.caption("Features are standardised, so distances are comparable; unscaled data would let large-range features dominate.")
    if name == "Decision Tree Classifier":
        depth = st.slider("Tree depth to draw", 1, 6, 3, key="clf_tree_depth")
        fig, ax = plt.subplots(figsize=(18, 7)); plot_tree(res["model"], max_depth=depth, feature_names=data["columns"], class_names=classes, filled=True, rounded=True, fontsize=7, ax=ax)
        ax.set_title(f"Decision tree (top {depth} levels; fitted depth = {res['model'].get_depth()})"); st.pyplot(fig); plt.close(fig)
    if name == "Logistic Regression":
        coef = pd.DataFrame(res["model"].coef_, index=classes, columns=data["columns"])
        st.plotly_chart(px.imshow(coef.round(2), color_continuous_scale="RdBu_r", aspect="auto", text_auto=True, title="Logistic Regression coefficients per class (scaled features)").update_layout(height=420), width="stretch")
    st.subheader("🔮 Classify a new person with this model")
    with st.form("clf_pred"):
        c = st.columns(4)
        g = c[0].selectbox("Gender", ["Female", "Male"]); age = c[1].number_input("Age", 10, 80, 25)
        h = c[2].number_input("Height (m)", 1.2, 2.2, 1.70, 0.01); w = c[3].number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5)
        c = st.columns(4)
        fh = c[0].selectbox("Family history of overweight", ["yes", "no"]); favc = c[1].selectbox("Frequent high-calorie food (FAVC)", ["yes", "no"])
        caec = c[2].selectbox("Snacking between meals (CAEC)", S.FREQ, index=1); calc = c[3].selectbox("Alcohol (CALC)", S.FREQ, index=1)
        c = st.columns(4)
        fcvc = c[0].slider("Vegetable frequency (FCVC)", 1.0, 3.0, 2.0, 0.1); ncp = c[1].slider("Main meals/day (NCP)", 1.0, 4.0, 3.0, 0.1)
        ch2o = c[2].slider("Water intake (CH2O)", 1.0, 3.0, 2.0, 0.1); faf = c[3].slider("Physical activity (FAF)", 0.0, 3.0, 1.0, 0.1)
        c = st.columns(4)
        tue = c[0].slider("Screen time (TUE)", 0.0, 2.0, 1.0, 0.1); smoke = c[1].selectbox("Smokes", ["no", "yes"]); scc = c[2].selectbox("Monitors calories (SCC)", ["no", "yes"])
        mt = c[3].selectbox("Transport (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        ok = st.form_submit_button("Classify")
    if ok:
        row = dict(Age=age, Gender=g, Height=h, Weight=w, CALC=calc, FAVC=favc, FCVC=fcvc, NCP=ncp, SCC=scc, SMOKE=smoke, CH2O=ch2o,
                   family_history_with_overweight=fh, FAF=faf, TUE=tue, CAEC=caec, MTRANS=mt)
        try:
            k, p = S.clf_predict_row(res, data, row)
            st.success(f"Predicted class: **{classes[k].replace('_', ' ')}** (BMI {w / h ** 2:.1f}; {name})")
            st.plotly_chart(px.bar(x=classes, y=p, labels={"x": "Class", "y": "Probability"}, title="Class probabilities", color_discrete_sequence=[PAL[0]]).update_layout(height=360), width="stretch")
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")


def _clf_compare(data, dkey):
    active = S.clf_active(); n = len(active)
    explain_clf_metrics()
    st.markdown(f"Trains **all {n} algorithms** on the same stratified split and builds the consolidated comparison table. 5-fold CV is added for the two best.")
    cmode = st.radio("Mode", ["Quick (default hyperparameters)", "Tuned (GridSearchCV for every model — slower)"], horizontal=True, key="clf_cmode")
    if st.button(f"▶ Run all {n} algorithms", type="primary", key="clf_all"):
        tune_all, rows, bar = cmode.startswith("Tuned"), [], st.progress(0.0)
        try:
            for i, (nm, sp) in enumerate(active.items()):
                bar.progress(i / n, text=f"Training {nm}…")
                r = c_clf_run(nm, json.dumps(S.build_params(sp["spec"], {}), default=str), dkey, tune_all, False, data)
                st.session_state.setdefault("clf_models", {})[nm] = r
                rows.append({"Part": sp["part"], "Algorithm": nm, **r["metrics"]})
            bar.progress(1.0, text="Cross-validating the two best models…")
            tbl = ranked_table(rows, "F1"); cvs = {}
            for nm in tbl["Algorithm"].head(2):
                cvs[nm] = c_clf_run(nm, json.dumps(S.build_params(S.CLF_SPECS[nm]["spec"], {}), default=str), dkey, tune_all, True, data)["cv"]
            tbl["CV F1 (5-fold)"] = tbl["Algorithm"].map(lambda x: float(cvs[x].mean()) if x in cvs else np.nan)
            st.session_state["clf_table"] = tbl; bar.empty()
        except Exception as e:  # noqa: BLE001
            bar.empty(); st.error(f"Comparison failed: {e}")
    tbl = st.session_state.get("clf_table")
    if tbl is None:
        return
    tbl = tbl[tbl["Algorithm"].isin(active)]
    st.dataframe(tbl.style.format({c: "{:.4f}" for c in tbl.columns if c not in ("Rank", "Part", "Algorithm")}, na_rep="—"), width="stretch", hide_index=True)
    long = tbl.melt(id_vars="Algorithm", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"], var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(long, x="Algorithm", y="Score", color="Metric", barmode="group", color_discrete_sequence=PAL, title="All metrics by algorithm").update_layout(height=450, xaxis_tickangle=-30, yaxis_range=[0, 1.02]), width="stretch")
    st.success(f"Best by weighted F1: **{tbl.iloc[0]['Algorithm']}** ({tbl.iloc[0]['F1']:.4f}). Open it from the dropdown and tune it to document the improvement.")
    st.download_button("⬇ Download comparison (CSV)", tbl.to_csv(index=False).encode(), "classification_comparison.csv", "text/csv")
