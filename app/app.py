"""Data Dynamos Lab - interactive Streamlit app for 23CSE301 Review 2 (K-Means + Agglomerative)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils as U

st.set_page_config(page_title="Data Dynamos Lab", page_icon="🚀", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#1f4e79,#2a9d8f);color:#fff;padding:1.2rem 1.6rem;border-radius:14px;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.8rem;color:#fff}.hero p{margin:.3rem 0 0;opacity:.92}
div[data-testid="stMetric"]{background:rgba(120,140,160,.12);border-radius:12px;padding:.7rem 1rem}
@media (max-width:700px){.hero h1{font-size:1.3rem}}
</style>""", unsafe_allow_html=True)
PAL = px.colors.qualitative.Safe


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ cached compute
@st.cache_data(show_spinner=False)
def c_load(source, uploaded_bytes, level):
    import io
    up = io.BytesIO(uploaded_bytes) if uploaded_bytes else None
    return U.load_source(source, up, level)


@st.cache_data(show_spinner=False)
def c_prep(table, features, scaler, drop_dups, labels):
    return U.preprocess(table, list(features), scaler, drop_dups, labels)


@st.cache_data(show_spinner=False)
def c_km_sweep(X, kmin, kmax, init, n_init, max_iter):
    return U.kmeans_sweep(X, kmin, kmax, init, n_init, max_iter)


@st.cache_data(show_spinner=False)
def c_ag_sweep(X, kmin, kmax):
    return U.agglo_sweep(X, kmin, kmax)


@st.cache_data(show_spinner=False)
def c_tsne(X):
    return U.tsne_2d(X)


def render_rubric():
    hero("Rubric Checklist", "How this project maps to the Review 2 clustering rubric (23CSE301).")
    chk = pd.DataFrame([
        ("B1 Implementation (4)", "K-Means and Agglomerative fitted, labels obtained", "Notebooks 11 & 12; pages K-Means / Agglomerative"),
        ("B2 Metrics (2)", "Silhouette, Davies-Bouldin, Calinski-Harabasz for all algorithms; Elbow + Dendrogram", "Metric cards, Compare page, elbow & dendrogram plots"),
        ("B3 Visualisation (2)", "PCA 2D scatter for every algorithm; t-SNE for at least one", "PCA on every page; t-SNE checkbox / notebook cells"),
        ("Regression (Review 1)", "10 algorithms, R²/RMSE/MAE, 5-fold CV for top 2, GridSearchCV, residual / predicted-vs-actual / importance plots", "Regression page (needs the box-office CSV)"),
        ("Classification (10 algorithms)", "Part A (1-5) + Part B (6-10), Accuracy/Precision/Recall/F1/ROC-AUC, confusion matrix, consolidated table", "Classification page"),
        ("D8 Bonus: GUI (+1)", "Web UI that accepts user input and returns model output", "This Streamlit app + 'Assign a new observation' predictor"),
        ("Bonus: Deployment (+1)", "Public URL", "Ready for Streamlit Community Cloud (see README); NOT deployed by this build"),
        ("Unsupervised rule", "Labels never used for fitting", "Regional Indicator only used for post-hoc ARI"),
    ], columns=["Rubric item", "Requirement", "Where implemented"])
    st.dataframe(chk, width="stretch", hide_index=True)
    st.caption("Full checklist with test status: RUBRIC_CHECKLIST.md in the repository.")


# ------------------------------------------------------------------ sidebar
import supervised_pages as SP

CLUSTER_VIEWS = ["Overview", "Dataset & Preprocessing", "Evaluation Metrics", "K-Means Clustering", "Agglomerative Hierarchical Clustering", "Compare clustering algorithms"]
CL_MAP = {"Overview": "Overview", "Dataset & Preprocessing": "Data & Preprocessing", "Evaluation Metrics": "Eval Metrics", "K-Means Clustering": "K-Means",
          "Agglomerative Hierarchical Clustering": "Agglomerative", "Compare clustering algorithms": "Compare Algorithms"}
st.sidebar.title("🚀 Data Dynamos Lab")
n_reg, n_clf = len(SP.S.REG_SPECS), len(SP.S.clf_active())
task = st.sidebar.radio("① Choose a task", ["Regression", "Classification", "Clustering"], key="task",
                        captions=[f"{n_reg} algorithms", f"{n_clf} algorithms", "2 algorithms"])
VIEWS = {"Regression": SP.reg_views(), "Classification": SP.clf_views(), "Clustering": CLUSTER_VIEWS}
view = st.sidebar.selectbox("② Choose an algorithm / view", VIEWS[task] + ["Rubric checklist"], key=f"view_{task}",
                            format_func=lambda v: "📋 Rubric checklist" if v == "Rubric checklist" else SP.view_label(task, v))
if view == "Rubric checklist":
    render_rubric()
    st.stop()
if task != "Clustering":
    st.sidebar.divider(); st.sidebar.subheader("Data settings")
    SP.render(task, view, hero)
    st.stop()
page = CL_MAP[view]
if page != "Eval Metrics":
    st.sidebar.divider()
    st.sidebar.subheader("Data settings")
    source = "Upload my own CSV"
    upload = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    level = "Country average (recommended)"
    if upload is None:
        hero("Clustering", "Upload a dataset to get started with clustering analysis.")
        st.info("⬅️ **Upload a CSV file** in the sidebar to begin clustering. Your CSV should have at least 10 rows and 2 numeric columns.")
        st.stop()
    try:
        table, region, unit_desc = c_load(source, upload.getvalue(), level)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load data: {e}")
        st.stop()

    num_cols = table.select_dtypes("number").columns.tolist()
    default_feats = num_cols
    features = st.sidebar.multiselect("Features used for clustering", table.columns.tolist(), default=default_feats,
                                      help="Labels such as Regional Indicator are never offered as inputs.")
    scaler_name = st.sidebar.selectbox("Scaler", list(U.SCALERS), help="Distance-based clustering needs comparable feature scales.")
    drop_dups = st.sidebar.checkbox("Drop duplicate rows", True)
    try:
        prep = c_prep(table, tuple(features), scaler_name, drop_dups, region)
    except Exception as e:  # noqa: BLE001
        st.error(f"Preprocessing failed: {e}")
        st.stop()
    X, labels_true = prep["X"], prep["labels"]
    kmax_allowed = int(min(12, len(X) - 1))


def scatter_fig(coords, lab, title, pca=None, centers=None, hover=None):
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "Cluster": [str(int(v)) for v in lab],
                       "Item": hover if hover is not None else range(len(lab))})
    fig = px.scatter(df, x="x", y="y", color="Cluster", hover_name="Item", color_discrete_sequence=PAL,
                     category_orders={"Cluster": sorted(df["Cluster"].unique(), key=int)}, title=title)
    if centers is not None:
        fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode="markers", name="Centroids",
                                 marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="white"))))
    xl, yl = ("PC1", "PC2") if pca is None else (f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var.)",
                                                 f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var.)")
    fig.update_layout(xaxis_title=xl, yaxis_title=yl, height=470, legend_title="Cluster", margin=dict(t=50, b=10))
    return fig


def size_fig(lab):
    s = pd.Series(lab).value_counts().sort_index()
    fig = px.bar(x=[str(i) for i in s.index], y=s.values, color=[str(i) for i in s.index], color_discrete_sequence=PAL,
                 labels={"x": "Cluster", "y": "Number of items"}, title="Cluster sizes", text=s.values)
    fig.update_layout(showlegend=False, height=380, margin=dict(t=50, b=10))
    return fig


def profile_fig(lab, title):
    z = pd.DataFrame(X, columns=prep["cols"]).groupby(np.asarray(lab)).mean()
    fig = px.imshow(z.round(2), color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto", text_auto=True,
                    labels=dict(x="Feature", y="Cluster", color="Mean (scaled)"), title=title)
    fig.update_layout(height=140 + 55 * len(z), margin=dict(t=50, b=10))
    return fig


def metric_row(m):
    a, b, c = st.columns(3)
    a.metric("Silhouette ↑", f"{m['silhouette']:.3f}", help="-1..1. Higher = tighter, better-separated clusters.")
    b.metric("Davies-Bouldin ↓", f"{m['davies_bouldin']:.3f}", help="Lower is better. Average similarity between each cluster and its closest rival.")
    c.metric("Calinski-Harabasz ↑", f"{m['calinski_harabasz']:.1f}", help="Higher is better. Between-cluster vs within-cluster dispersion ratio.")


def explain_metrics():
    with st.expander("📊 Evaluation Metrics – What do these metrics mean?"):
        st.markdown("### Clustering Evaluation Metrics")
        st.markdown("""
| Metric | Range | Ideal | Description |
|--------|-------|-------|-------------|
| **Silhouette Score** | −1 to 1 | Higher (→ 1) | Compares each point's average distance to its own cluster vs. the nearest other cluster. Above ~0.5 = strong; 0.25–0.5 = moderate; near 0 = overlapping. |
| **Davies-Bouldin Index** | ≥ 0 | Lower (→ 0) | Average ratio of within-cluster scatter to between-cluster separation for each cluster's worst-case neighbour. |
| **Calinski-Harabasz Index** | > 0 | Higher | Ratio of between-cluster to within-cluster variance. Tends to grow with k, so compare at similar k. |
| **Inertia** *(K-Means only)* | ≥ 0 | Lower | Sum of squared distances to the nearest centroid. Always falls as k grows → use the *elbow* heuristic. |
""")


def show_validation(lab):
    if labels_true is not None:
        v = U.ari(labels_true.to_numpy(), lab)
        with st.expander("Post-hoc validation against Regional Indicator (not used for fitting)"):
            st.metric("Adjusted Rand Index", f"{v:.3f}", help="0 = random agreement, 1 = identical to the regions.")


def need_k_range(key):
    c1, c2 = st.columns(2)
    kmin = c1.number_input("Minimum k", 2, kmax_allowed - 1, 2, key=key + "min")
    kmx = c2.number_input("Maximum k", int(kmin) + 1, kmax_allowed, min(10, kmax_allowed), key=key + "max")
    return int(kmin), int(kmx)


# ------------------------------------------------------------------ pages
if page == "Eval Metrics":
    hero("Clustering · 📊 Evaluation Metrics", "Understanding the metrics used to evaluate clustering models.")
    explain_metrics()
    st.stop()

if page == "Overview":
    hero("Clustering · Overview", "K-Means and Agglomerative Hierarchical Clustering on the World Happiness Report, computed live.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows after cleaning", prep["report"]["rows_out"]); c2.metric("Features (after encoding)", prep["report"]["cols_out"])
    c3.metric("Scaler", scaler_name)
    st.markdown("""
### How to use the clustering section
1. **Dataset & Preprocessing**: inspect the data and what was cleaned.
2. **K-Means Clustering**: pick the k range, see the elbow curve and silhouette scores, choose k, inspect clusters, and assign a new country to a cluster.
3. **Agglomerative Hierarchical Clustering**: dendrogram and ward / complete / average / single comparison.
4. **Compare clustering algorithms**: all models side by side at the same k.

Every number and chart is produced by running scikit-learn/SciPy on your current settings. Change anything in the sidebar and the results recompute. Class labels are never used for fitting.
""")
    explain_metrics()

elif page == "Data & Preprocessing":
    hero("Dataset & Preprocessing", f"Unit of analysis: {unit_desc}")
    rep = prep["report"]
    a, b, c, d = st.columns(4)
    a.metric("Rows in", rep["rows_in"]); b.metric("Duplicates removed", rep["duplicates_removed"])
    c.metric("Missing cells imputed", rep["missing_filled"]); d.metric("Final shape", f"{rep['rows_out']} × {rep['cols_out']}")
    t1, t2, t3, t4 = st.tabs(["Preview", "Quality audit", "Correlation", "Scaling effect"])
    with t1:
        st.dataframe(table.head(25), width="stretch")
    with t2:
        audit = pd.DataFrame({"dtype": table.dtypes.astype(str), "missing": table.isna().sum(),
                              "missing %": (table.isna().mean() * 100).round(1), "unique": table.nunique()})
        st.dataframe(audit, width="stretch")
        st.dataframe(table.describe().T.round(3), width="stretch")
        st.caption(f"Strategy: numeric → median imputation, categorical → mode + one-hot ({rep['one_hot_encoded'] or 'none'}); "
                   f"then {scaler_name}. Outliers are kept (they are real observations).")
    with t3:
        corr = prep["df"].corr()
        st.plotly_chart(px.imshow(corr.round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True,
                                  title="Feature correlation heatmap", aspect="auto").update_layout(height=550), width="stretch")
        st.caption("Pairs with |r| > 0.9 are redundant; deselect one in the sidebar if you see any.")
    with t4:
        col = st.selectbox("Feature", prep["cols"])
        i = prep["cols"].index(col)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=prep["df"][col], name="Original", opacity=.65))
        fig.add_trace(go.Histogram(x=X[:, i], name=f"After {scaler_name}", opacity=.65, xaxis="x2"))
        fig.update_layout(barmode="overlay", title=f"{col}: before vs after scaling", xaxis=dict(title="Original scale"),
                          xaxis2=dict(title="Scaled", overlaying="x", side="top"), yaxis_title="Count", height=400)
        st.plotly_chart(fig, width="stretch")

elif page == "K-Means":
    hero("K-Means Clustering", "Elbow curve, silhouette comparison, final clusters and live prediction.")
    explain_metrics()
    with st.form("kmeans_form"):
        st.subheader("Parameters")
        kmin, kmx = need_k_range("km")
        c1, c2, c3 = st.columns(3)
        init = c1.selectbox("Initialisation", ["k-means++", "random"])
        n_init = c2.slider("n_init (restarts)", 1, 30, 10)
        max_iter = c3.slider("max_iter", 50, 1000, 300, step=50)
        k_choice = st.slider("Number of clusters (k) for the final model", 2, kmax_allowed, min(3, kmax_allowed))
        go_btn = st.form_submit_button("▶ Run K-Means", type="primary")
    if go_btn or "km_ran" in st.session_state:
        st.session_state["km_ran"] = True
        try:
            with st.spinner("Fitting K-Means for each k…"):
                sweep = c_km_sweep(X, kmin, kmx, init, n_init, max_iter)
            k_sil = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
            if not (kmin <= k_choice <= kmx):
                st.warning(f"Final k={k_choice} is outside the scanned range {kmin}-{kmx}; the sweep plots just won't mark it.")
            st.subheader("Choosing k")
            a, b = st.columns(2)
            f = px.line(sweep, x="k", y="inertia", markers=True, title="Elbow curve (inertia vs k)", labels={"inertia": "Inertia (SSE)", "k": "k"})
            f.add_vline(x=k_choice, line_dash="dash", annotation_text=f"selected k={k_choice}")
            a.plotly_chart(f.update_layout(height=380), width="stretch")
            f = px.bar(sweep, x="k", y="silhouette", title="Silhouette score by k", labels={"silhouette": "Silhouette"})
            f.add_vline(x=k_sil, line_dash="dot", annotation_text=f"best k={k_sil}")
            b.plotly_chart(f.update_layout(height=380), width="stretch")
            st.info(f"Highest silhouette in the scanned range: **k = {k_sil}** ({sweep['silhouette'].max():.3f}). "
                    "The elbow is a visual judgement; compare it with the silhouette and your domain needs.")
            st.dataframe(sweep.round(3).set_index("k"), width="stretch")

            with st.spinner("Fitting final model…"):
                km = U.kmeans_fit(X, k_choice, init, n_init, max_iter)
                m = U.metrics(X, km.labels_)
                pca, P = U.pca_2d(X)
            st.subheader(f"Final model (k = {k_choice})")
            metric_row(m)
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(scatter_fig(P, km.labels_, "K-Means clusters in PCA space", pca, pca.transform(km.cluster_centers_), list(prep["df"].index)), width="stretch")
            c2.plotly_chart(size_fig(km.labels_), width="stretch")
            st.plotly_chart(profile_fig(km.labels_, "Cluster profile (scaled feature means)"), width="stretch")
            if st.checkbox("Also show t-SNE (slower)", key="km_tsne"):
                with st.spinner("Running t-SNE…"):
                    st.plotly_chart(scatter_fig(c_tsne(X), km.labels_, "K-Means clusters in t-SNE space", None, None, list(prep["df"].index)).update_layout(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2"), width="stretch")
            out = prep["df"].copy(); out["cluster"] = km.labels_
            st.dataframe(out.groupby("cluster").agg(["mean"]).round(2).droplevel(1, axis=1).assign(n=out["cluster"].value_counts().sort_index()), width="stretch")
            with st.expander("Browse items in each cluster"):
                cl = st.selectbox("Cluster", sorted(set(km.labels_)))
                st.write(", ".join(map(str, out.index[out["cluster"] == cl])))
            st.download_button("⬇ Download labelled data (CSV)", out.to_csv().encode(), "kmeans_clusters.csv", "text/csv")
            show_validation(km.labels_)

            st.subheader("🔮 Assign a new observation to a cluster")
            st.caption("Enter feature values; they are transformed with the fitted scaler and assigned by the trained K-Means model.")
            raw_df = prep["df"]
            with st.form("predict_form"):
                cols = st.columns(3); vals = {}
                for i, cname in enumerate(prep["cols"]):
                    lo, hi, med = float(raw_df[cname].min()), float(raw_df[cname].max()), float(raw_df[cname].median())
                    rng = (hi - lo) or 1.0
                    vals[cname] = cols[i % 3].number_input(cname, value=med, min_value=lo - rng, max_value=hi + rng, step=rng / 100, format="%.3f")
                pred_btn = st.form_submit_button("Predict cluster")
            if pred_btn:
                xr = prep["scaler"].transform(pd.DataFrame([vals])[prep["cols"]].to_numpy(float))
                cl = int(km.predict(xr)[0]); dist = np.linalg.norm(km.cluster_centers_ - xr, axis=1)
                st.success(f"Assigned to **Cluster {cl}** (distance to its centroid: {dist[cl]:.2f}).")
                st.dataframe(pd.DataFrame({"cluster": range(len(dist)), "distance to centroid": dist.round(3)}).set_index("cluster").T, width="stretch")
        except Exception as e:  # noqa: BLE001
            st.error(f"K-Means failed: {e}")
    else:
        st.info("Set the parameters above and press **Run K-Means**.")

elif page == "Agglomerative":
    hero("Agglomerative Hierarchical Clustering", "Dendrogram and linkage comparison (ward, complete, average, single).")
    explain_metrics()
    with st.expander("ℹ️ Linkage strategies"):
        st.markdown("""
- **ward:** merges the pair that least increases within-cluster variance → compact, balanced clusters (Euclidean only).
- **complete:** max pairwise distance → compact clusters, sensitive to outliers.
- **average:** mean pairwise distance → a compromise.
- **single:** min pairwise distance → can follow elongated shapes but tends to *chain* into one giant cluster plus tiny ones.""")
    with st.form("agg_form"):
        c1, c2, c3 = st.columns(3)
        link = c1.selectbox("Linkage for dendrogram & plots", U.LINKAGES)
        k = c2.slider("Number of clusters (k)", 2, kmax_allowed, min(3, kmax_allowed))
        p = c3.slider("Dendrogram leaves shown (truncation)", 10, 60, 30)
        kmin, kmx = need_k_range("ag")
        run = st.form_submit_button("▶ Run Agglomerative", type="primary")
    if run or "ag_ran" in st.session_state:
        st.session_state["ag_ran"] = True
        try:
            with st.spinner("Building linkage tree…"):
                Z, d = U.dendrogram_data(X, link, p)
            fig = go.Figure()
            for ic, dc in zip(d["icoord"], d["dcoord"]):
                fig.add_trace(go.Scatter(x=ic, y=dc, mode="lines", line=dict(color=PAL[0]), hoverinfo="y", showlegend=False))
            h = U.cut_height(Z, k)
            if h is not None:
                fig.add_hline(y=h, line_dash="dash", line_color="crimson", annotation_text=f"cut → {k} clusters")
            fig.update_xaxes(tickvals=[5 + 10 * i for i in range(len(d["ivl"]))], ticktext=d["ivl"], title="Leaf (count of items in brackets for merged groups)")
            fig.update_layout(title=f"Dendrogram ({link} linkage, last {p} merges)", yaxis_title="Merge distance", height=450)
            st.plotly_chart(fig, width="stretch")

            with st.spinner("Comparing linkage strategies…"):
                sw = c_ag_sweep(X, kmin, kmx)
            st.subheader("Silhouette across k and linkage")
            f = px.line(sw, x="k", y="silhouette", color="linkage", markers=True, color_discrete_sequence=PAL, labels={"silhouette": "Silhouette"})
            st.plotly_chart(f.update_layout(height=380), width="stretch")
            st.subheader(f"Linkage comparison at k = {k}")
            rows, labs = [], {}
            for mth in U.LINKAGES:
                labs[mth] = U.agglo_labels(X, k, mth); cnt = np.bincount(labs[mth])
                rows.append(dict(linkage=mth, sizes=str(cnt.tolist()), smallest_cluster_pct=round(100 * cnt.min() / len(X), 1), **U.metrics(X, labs[mth])))
            cmp_df = pd.DataFrame(rows).set_index("linkage")
            st.dataframe(cmp_df.round(3), width="stretch")
            weak = [m for m in U.LINKAGES if cmp_df.loc[m, "smallest_cluster_pct"] < 5]
            if weak:
                st.warning(f"{', '.join(weak)}: smallest cluster is under 5% of the data. A high silhouette here can reflect outliers split off (chaining), not good structure.")
            st.subheader(f"Selected linkage: {link}")
            metric_row(U.metrics(X, labs[link]))
            pca, P = U.pca_2d(X)
            a, b = st.columns([3, 2])
            a.plotly_chart(scatter_fig(P, labs[link], f"{link} clusters in PCA space", pca, None, list(prep["df"].index)), width="stretch")
            b.plotly_chart(size_fig(labs[link]), width="stretch")
            st.plotly_chart(profile_fig(labs[link], "Cluster profile (scaled feature means)"), width="stretch")
            if st.checkbox("Also show PCA for all four linkages"):
                cols = st.columns(2)
                for i, mth in enumerate(U.LINKAGES):
                    cols[i % 2].plotly_chart(scatter_fig(P, labs[mth], f"{mth}", pca, None, list(prep["df"].index)).update_layout(height=380), width="stretch")
            if st.checkbox("Also show t-SNE (slower)", key="ag_tsne"):
                with st.spinner("Running t-SNE…"):
                    st.plotly_chart(scatter_fig(c_tsne(X), labs[link], f"{link} clusters in t-SNE space", None, None, list(prep["df"].index)).update_layout(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2"), width="stretch")
            out = prep["df"].copy(); out["cluster"] = labs[link]
            st.download_button("⬇ Download labelled data (CSV)", out.to_csv().encode(), f"agglomerative_{link}_clusters.csv", "text/csv")
            show_validation(labs[link])
        except Exception as e:  # noqa: BLE001
            st.error(f"Agglomerative clustering failed: {e}")
    else:
        st.info("Set the parameters above and press **Run Agglomerative**.")

elif page == "Compare Algorithms":
    hero("Compare Algorithms", "K-Means and all four linkages, evaluated with the three mandatory metrics at the same k.")
    k = st.slider("Number of clusters (k)", 2, kmax_allowed, min(3, kmax_allowed), key="cmp_k")
    try:
        with st.spinner("Fitting all models…"):
            res = {"K-Means": U.kmeans_fit(X, k).labels_}
            for mth in U.LINKAGES:
                res[f"Agglomerative ({mth})"] = U.agglo_labels(X, k, mth)
            tbl = pd.DataFrame({n: {**U.metrics(X, l), "smallest_cluster_pct": 100 * np.bincount(l).min() / len(X)} for n, l in res.items()}).T
        st.dataframe(tbl.round(3).rename_axis("model"), width="stretch")
        a, b, c = st.columns(3)
        for col, key, title in [(a, "silhouette", "Silhouette ↑"), (b, "davies_bouldin", "Davies-Bouldin ↓"), (c, "calinski_harabasz", "Calinski-Harabasz ↑")]:
            col.plotly_chart(px.bar(tbl.reset_index(), x="index", y=key, title=title, labels={"index": ""}, color="index", color_discrete_sequence=PAL).update_layout(showlegend=False, height=360, xaxis_tickangle=-35), width="stretch")
        ok = tbl[tbl["smallest_cluster_pct"] >= 5]
        if len(ok):
            st.success(f"Highest silhouette among models without a degenerate (<5%) cluster: **{ok['silhouette'].idxmax()}** ({ok['silhouette'].max():.3f}).")
        pca, P = U.pca_2d(X)
        pick = st.multiselect("Show PCA plots for", list(res), default=["K-Means", "Agglomerative (ward)"])
        cols = st.columns(2)
        for i, n in enumerate(pick):
            cols[i % 2].plotly_chart(scatter_fig(P, res[n], n, pca, None, list(prep["df"].index)).update_layout(height=400), width="stretch")
        explain_metrics()
    except Exception as e:  # noqa: BLE001
        st.error(f"Comparison failed: {e}")

else:
    render_rubric()
