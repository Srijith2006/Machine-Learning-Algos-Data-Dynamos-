"""TEST FIXTURE ONLY: synthetic data with the same column schema as enhanced_box_office_data_2000-2024.csv.
It exists so the regression code path can be tested when the real CSV is unavailable. Never use for reported results."""
import numpy as np, pandas as pd, sys
def make(n=3000, seed=0):
    r = np.random.default_rng(seed)
    year = r.integers(2000, 2025, n); votes = np.exp(r.normal(8.5, 1.6, n)).astype(int)
    rating = np.clip(r.normal(6.4, 1.0, n), 1, 9.5)
    gen = ["Action","Drama","Comedy","Horror","Adventure","Animation","Thriller","Romance","Sci-Fi"]
    genres = [",".join(r.choice(gen, r.integers(1, 4), replace=False)) for _ in range(n)]
    ctry = [",".join(r.choice(["United States","United Kingdom","France","China","India","Japan"], r.integers(1, 3), replace=False)) for _ in range(n)]
    ww = np.exp(0.9*np.log(votes) + 0.5*(rating-6) + r.normal(6.5, 1.0, n)).astype(float)
    dom = ww * r.uniform(.2, .7, n)
    df = pd.DataFrame({"Rank": np.arange(1, n+1), "Release Group": r.choice(list("ABCDEFGH"), n), "Year": year,
        "$Worldwide": ww.round(0), "$Domestic": dom.round(0), "Domestic %": (dom/ww*100).round(1),
        "$Foreign": (ww-dom).round(0), "Foreign %": ((ww-dom)/ww*100).round(1),
        "Rating": [f"{x:.3f}/10" for x in rating], "Vote_Count": votes, "Genres": genres,
        "Production_Countries": ctry, "Original_Language": r.choice(["en","fr","zh","hi","ja","es","de","ko","it","pt"], n, p=[.6,.06,.06,.06,.05,.05,.04,.04,.02,.02])})
    df.loc[r.choice(n, 40, replace=False), "Vote_Count"] = np.nan
    df.loc[r.choice(n, 40, replace=False), "Genres"] = np.nan
    df.loc[r.choice(n, 30, replace=False), "Rating"] = np.nan
    return df
if __name__ == "__main__":
    make().to_csv(sys.argv[1], index=False); print("wrote", sys.argv[1])
