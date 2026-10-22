"""Data loading, preprocessing and clustering helpers (real sklearn/scipy execution, no mock values)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "world_happiness_report.csv"
BUILTIN = "World Happiness Report (included)"
LINKAGES = ["ward", "complete", "average", "single"]
SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}


def _mode(s):
    return s.mode().iloc[0] if s.notna().any() else "Unknown"


def load_source(source, uploaded=None, level="Country average (recommended)"):
    """Return (table, labels|None, id_description). Labels are for post-hoc validation only."""
    if source == BUILTIN:
        raw = pd.read_csv(DATA_PATH)
        feats = [c for c in raw.columns if c not in ("Country Name", "Regional Indicator", "Year")]
        region = raw.groupby("Country Name")["Regional Indicator"].agg(_mode)
        if level.startswith("Country average"):
            table = raw.groupby("Country Name")[feats].mean()
            return table, region.loc[table.index], "one row per country (mean over 2005-2022)"
        table = raw[feats].copy()
        table.index = raw["Country Name"] + " " + raw["Year"].astype(str)
        return table, raw["Country Name"].map(region).set_axis(table.index), "one row per country-year"
    if uploaded is None:
        raise ValueError("Upload a CSV file to continue.")
    df = pd.read_csv(uploaded)
    if df.shape[0] < 10 or df.shape[1] < 2:
        raise ValueError("The CSV needs at least 10 rows and 2 columns.")
    return df, None, "uploaded CSV (row index)"


def preprocess(table, features, scaler="StandardScaler", drop_dups=True, labels=None):
    if not features:
        raise ValueError("Select at least one feature.")
    d = table[features].copy()
    lab = None if labels is None else pd.Series(np.asarray(labels), index=d.index)
    rep = {"rows_in": len(d)}
    if drop_dups:
        keep = ~d.duplicated().to_numpy()
        rep["duplicates_removed"] = int((~keep).sum())
        d = d[keep]; lab = None if lab is None else lab[keep]
    else:
        rep["duplicates_removed"] = 0
    num = d.select_dtypes("number").columns.tolist()
    cat = [c for c in d.columns if c not in num]
    rep["missing_filled"] = int(d.isna().sum().sum())
    for c in num:
        d[c] = d[c].fillna(d[c].median())
    for c in cat:
        d[c] = d[c].fillna(_mode(d[c]) if d[c].notna().any() else "missing")
    if cat:
        d = pd.get_dummies(d, columns=cat, dtype=float)
    rep["one_hot_encoded"] = cat
    if len(d) < 10:
        raise ValueError("Fewer than 10 rows remain after cleaning.")
    sc = SCALERS[scaler]()
    X = sc.fit_transform(d.to_numpy(float))
    rep["rows_out"], rep["cols_out"] = X.shape
    return dict(X=X, df=d, labels=lab, scaler=sc, cols=d.columns.tolist(), report=rep)


def metrics(X, labels):
    n = len(set(labels))
    if n < 2 or n >= len(X):
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(silhouette=silhouette_score(X, labels), davies_bouldin=davies_bouldin_score(X, labels),
                calinski_harabasz=calinski_harabasz_score(X, labels))


def kmeans_sweep(X, kmin, kmax, init="k-means++", n_init=10, max_iter=300):
    rows = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=SEED).fit(X)
        rows.append(dict(k=k, inertia=km.inertia_, **metrics(X, km.labels_)))
    return pd.DataFrame(rows)


def kmeans_fit(X, k, init="k-means++", n_init=10, max_iter=300):
    return KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=SEED).fit(X)


def agglo_labels(X, k, method):
    return AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)


def agglo_sweep(X, kmin, kmax):
    rows = []
    for m in LINKAGES:
        for k in range(kmin, kmax + 1):
            lab = agglo_labels(X, k, m)
            rows.append(dict(linkage=m, k=k, min_cluster_share=np.bincount(lab).min() / len(X), **metrics(X, lab)))
    return pd.DataFrame(rows)


def dendrogram_data(X, method, p=30):
    Z = linkage(X, method=method)
    d = dendrogram(Z, truncate_mode="lastp", p=p, no_plot=True)
    return Z, d


def cut_height(Z, k):
    """Height of a horizontal cut that yields k clusters."""
    if k < 2 or k > len(Z):
        return None
    return float((Z[-(k - 1), 2] + Z[-k, 2]) / 2)


def pca_2d(X):
    p = PCA(n_components=2, random_state=SEED).fit(X)
    return p, p.transform(X)


def tsne_2d(X):
    perp = max(5, min(30, (len(X) - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=SEED).fit_transform(X)


def ari(labels_true, labels_pred):
    m = pd.Series(labels_true).notna().to_numpy() & (pd.Series(labels_true).astype(str) != "Unknown").to_numpy()
    if m.sum() < 2:
        return np.nan
    return adjusted_rand_score(np.asarray(labels_true)[m], np.asarray(labels_pred)[m])
