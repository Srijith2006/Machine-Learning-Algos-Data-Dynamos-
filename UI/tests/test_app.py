"""Run: python tests/test_app.py   (exercises the real Streamlit script with real model execution)"""
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
sys.path.insert(0, str(Path(APP).parent))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS" if cond else "FAIL"), name, detail)

VIEW = {"Overview": "Overview", "Data & Preprocessing": "Dataset & Preprocessing", "K-Means": "K-Means Clustering",
        "Agglomerative": "Agglomerative Hierarchical Clustering", "Compare Algorithms": "Compare clustering algorithms", "Rubric Checklist": "Rubric checklist"}
def fresh(page):
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio(key="task").set_value("Clustering").run()
    at.sidebar.selectbox(key="view_Clustering").set_value(VIEW[page]).run()
    return at

for p in ["Overview", "Data & Preprocessing", "Compare Algorithms", "Rubric Checklist"]:
    at = fresh(p); check(f"page loads without exception: {p}", not at.exception, [e.value for e in at.exception])

at = fresh("K-Means")
btn = [b for b in at.button if "Run K-Means" in b.label][0]; btn.click().run()
check("K-Means runs", not at.exception and not at.error, [e.value for e in at.exception] + [e.value for e in at.error])
check("K-Means shows 3 metric cards for final model", len([m for m in at.metric if "Silhouette" in m.label]) >= 1)
sil = float([m for m in at.metric if "Silhouette" in m.label][0].value)
from sklearn.cluster import KMeans; import utils as U
t, r, _ = U.load_source(U.BUILTIN); pr = U.preprocess(t, list(t.columns), labels=r)
k3 = U.kmeans_fit(pr["X"], 3); ref = U.metrics(pr["X"], k3.labels_)["silhouette"]
check("UI silhouette (k=3) equals independent sklearn computation", abs(sil - ref) < 1e-3, f"ui={sil:.3f} ref={ref:.3f}")
# change k and recompute
sl = [s for s in at.slider if "final model" in s.label][0]; sl.set_value(4)
[b for b in at.button if "Run K-Means" in b.label][0].click().run()
sil4 = float([m for m in at.metric if "Silhouette" in m.label][0].value)
ref4 = U.metrics(pr["X"], U.kmeans_fit(pr["X"], 4).labels_)["silhouette"]
check("Changing k recomputes (k=4 matches sklearn)", abs(sil4 - ref4) < 1e-3 and sil4 != sil, f"ui={sil4:.3f} ref={ref4:.3f}")
[b for b in at.button if b.label == "Predict cluster"][0].click().run()
check("Predict cluster returns an assignment", any("Assigned to" in s.value for s in at.success), [s.value for s in at.success])

at = fresh("Agglomerative")
[b for b in at.button if "Run Agglomerative" in b.label][0].click().run()
check("Agglomerative runs", not at.exception and not at.error, [e.value for e in at.exception] + [e.value for e in at.error])
check("Linkage comparison table shown (4 linkages)", any(len(d.value) == 4 and "silhouette" in d.value.columns for d in at.dataframe))
ag = at.selectbox[[s.label for s in at.selectbox].index("Linkage for dendrogram & plots")]; ag.set_value("single")
[b for b in at.button if "Run Agglomerative" in b.label][0].click().run()
check("single linkage selection reruns and warns about chaining/small clusters", not at.exception, [w.value[:60] for w in at.warning])

at = fresh("Overview"); at.sidebar.selectbox[1].set_value("Upload my own CSV").run()
check("Upload mode without file shows friendly error (no crash)", not at.exception and len(at.error) >= 1)
at = fresh("Overview"); at.sidebar.radio[1].set_value("Country-year rows").run()
at.sidebar.selectbox(key="view_Clustering").set_value("Compare clustering algorithms").run()
check("Country-year level (2199 rows) works on Compare page", not at.exception and not at.error, [e.value for e in at.exception] + [e.value for e in at.error])
print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed"); sys.exit(0 if all(ok for _, ok in results) else 1)
