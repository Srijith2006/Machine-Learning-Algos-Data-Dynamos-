# Grouped UI update

Copy `app/` and `tests/` over your project (overwrite). `app/utils.py` is unchanged, included for completeness.
Then: `streamlit run app/app.py`

Sidebar: (1) choose a task: Regression / Classification / Clustering; (2) choose an algorithm from the numbered dropdown.
Every dropdown also has Dataset & EDA (or Overview), Compare all, and a Rubric checklist.

Switches: `INCLUDE_PART_B = True` in `app/supervised.py` shows classification algorithms 6-10 (set False to hide them).
Regression needs `data/enhanced_box_office_data_2000-2024.csv` (your file; or upload it in the sidebar).
