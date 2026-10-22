"""Run: python tests/test_supervised_ui.py -- drives the real grouped UI (task radio -> algorithm dropdown).
Regression uses a SYNTHETIC fixture (the real box-office CSV was not supplied). Classification uses real data and is compared
with the notebooks' saved Part A results (numbers hard-coded below)."""
import os, sys, tempfile
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parent.parent; APP = str(ROOT / "app" / "app.py")
sys.path.insert(0, str(ROOT / "app")); sys.path.insert(0, str(ROOT / "tests"))
import make_box_office_fixture as F, supervised as S, supervised_pages as SP
res = []
def check(n, c, d=""): res.append(bool(c)); print("PASS" if c else "FAIL", n, d)
def errs(at): return [e.value for e in at.exception] + [e.value for e in at.error]
def metric(at, label): return float([m for m in at.metric if m.label == label][0].value.replace(",", ""))
def nav(at, task, view):
    if at.sidebar.radio(key="task").value != task: at.sidebar.radio(key="task").set_value(task).run()
    at.sidebar.selectbox(key=f"view_{task}").set_value(view).run(); return at
def new(): return AppTest.from_file(APP, default_timeout=900).run()

# ---- grouping / naming
at = new()
check("Task radio has exactly 3 groups", at.sidebar.radio(key="task").options == ["Regression", "Classification", "Clustering"])
check("Regression dropdown: EDA + 10 algorithms + Compare + Rubric", len(at.sidebar.selectbox(key="view_Regression").options) == 13)
at = nav(at, "Classification", "Dataset & EDA"); check("Classification dropdown: EDA + 10 algorithms + Compare + Rubric", len(at.sidebar.selectbox(key="view_Classification").options) == 13)
at = nav(at, "Clustering", "Overview"); check("Clustering dropdown: 5 views + Rubric", len(at.sidebar.selectbox(key="view_Clustering").options) == 6)
check("Rubric names used (spot check)", all(n in S.CLF_SPECS for n in ["Decision Tree Classifier", "Random Forest Classifier", "AdaBoost Classifier", "Gradient Boosting Classifier", "Bagging Classifier", "MLP Classifier (Neural Network)"]))
check("Numbered labels", SP.view_label("Regression", "K-Nearest Neighbors Regressor") == "10. K-Nearest Neighbors Regressor" and SP.view_label("Classification", "Bagging Classifier") == "[Part B] 9. Bagging Classifier")
S.INCLUDE_PART_B = False; at = nav(new(), "Classification", "Dataset & EDA"); check("Part B switch OFF hides algos 6-10", len(at.sidebar.selectbox(key="view_Classification").options) == 8); S.INCLUDE_PART_B = True

# ---- regression (missing CSV, then fixture)
os.environ["BOX_OFFICE_CSV"] = "/nonexistent/file.csv"
at = nav(new(), "Regression", "Linear Regression"); check("Regression: missing CSV -> warning, no crash", not at.exception and len(at.warning) >= 1)
fx = Path(tempfile.mkdtemp()) / "box.csv"; F.make(3000).to_csv(fx, index=False); os.environ["BOX_OFFICE_CSV"] = str(fx)
data = S.reg_prepare(F.make(3000))
at = nav(new(), "Regression", "Dataset & EDA"); check("Regression: EDA view loads", not at.exception and not at.error, errs(at))
nav(at, "Regression", "Ridge Regression"); at.number_input(key="reg_Ridge Regression_alpha").set_value(10.0).run(); at.button(key="reg_go").click().run()
ref = S.reg_run("Ridge Regression", {"alpha": 10.0}, data)["metrics"]
check("Regression: Ridge UI R² (log) equals direct computation", not at.exception and abs(metric(at, "R² (log scale)") - ref["R2_log"]) < 1e-3, f"{metric(at,'R² (log scale)'):.4f} vs {ref['R2_log']:.4f}")
at.radio(key="reg_mode_Ridge Regression").set_value("Tune with GridSearchCV (5-fold)").run(); at.button(key="reg_go").click().run()
check("Regression: GridSearchCV tuning runs", not at.exception and not at.error and "best_params" in at.session_state["reg_models"]["Ridge Regression"], errs(at))
for n in S.REG_SPECS:
    nav(at, "Regression", n); at.button(key="reg_go").click().run()
    check(f"Regression UI trains: {n}", not at.exception and not at.error and n in at.session_state["reg_models"], errs(at))
nav(at, "Regression", "Polynomial Regression"); at.button[[b.label for b in at.button].index("Predict worldwide gross")].click().run()
check("Regression: prediction returned", any("Predicted worldwide gross" in s.value for s in at.success), errs(at))
nav(at, "Regression", "Compare all algorithms"); at.button(key="reg_all").click().run(); t = at.session_state["reg_table"]
check("Regression: compare table 10 ranked + CV for top 2", not at.exception and len(t) == 10 and t["CV R² (5-fold, log)"].notna().sum() == 2 and t["R2"].is_monotonic_decreasing, errs(at))

# ---- classification (real data)
at = nav(new(), "Classification", "Dataset & EDA"); check("Classification: EDA view loads", not at.exception and not at.error, errs(at))
def train(name, tune=False):
    nav(at, "Classification", name)
    if tune: at.radio(key=f"clf_mode_{name}").set_value("Tune with GridSearchCV (5-fold)").run()
    at.button(key="clf_go").click().run()
train("Decision Tree Classifier", True); check("Classification: tuned Decision Tree == notebook (0.9811)", abs(metric(at, "Accuracy") - 0.9811) < 5e-5, f"{metric(at,'Accuracy'):.4f}")
train("Support Vector Machine (SVC)", True); check("Classification: tuned SVC == notebook (0.9764)", abs(metric(at, "Accuracy") - 0.9764) < 5e-5, f"{metric(at,'Accuracy'):.4f}")
train("K-Nearest Neighbors", True); check("Classification: tuned KNN == notebook (0.9243 / AUC 0.9864)", abs(metric(at, "Accuracy") - 0.9243) < 5e-5 and abs(metric(at, "ROC-AUC (OvR)") - 0.9864) < 5e-5)
train("Naive Bayes (Gaussian)"); check("Classification: default Naive Bayes == notebook (0.7447)", abs(metric(at, "Accuracy") - 0.7447) < 5e-5)
train("Logistic Regression"); check("Classification: default LogReg == notebook (0.9078)", abs(metric(at, "Accuracy") - 0.9078) < 5e-5)
for n in S.clf_active():
    train(n); check(f"Classification UI trains: {n} [Part {S.CLF_SPECS[n]['part']}]", not at.exception and not at.error and n in at.session_state["clf_models"], errs(at))
nav(at, "Classification", "MLP Classifier (Neural Network)"); at.button[[b.label for b in at.button].index("Classify")].click().run()
check("Classification: new-person prediction returned", any("Predicted class" in s.value for s in at.success), errs(at))
nav(at, "Classification", "Compare all algorithms"); at.button(key="clf_all").click().run(); t = at.session_state["clf_table"]
check("Classification: consolidated table = 10 algorithms, Parts A+B, CV for top 2", not at.exception and len(t) == 10 and set(t["Part"]) == {"A", "B"} and t["CV F1 (5-fold)"].notna().sum() == 2, errs(at))
print(f"\n{sum(res)}/{len(res)} checks passed"); sys.exit(0 if all(res) else 1)
