"""
Reproduce Table 3 of "Isolation Forest" (Liu, Ting & Zhou, ICDM 2008)
using scikit-learn's IsolationForest (no own implementation).

Usage:
    python reproduce_sklearn.py                 # all datasets, 10 runs each
    python reproduce_sklearn.py Breastw Pima    # only some datasets
    python reproduce_sklearn.py --runs 3
    python reproduce_sklearn.py --out my_results.csv

Prints the comparison table and also saves it to results.csv.

Needs: numpy, scikit-learn   (pip install numpy scikit-learn)
"""
import argparse
import csv
import os
import time
import urllib.request

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# ---- parameters: the paper's settings (Sec. 4.1 and 5) ----------------------
N_TREES = 100        # t   : number of trees
SUBSAMPLE = 256      # psi : points used to build each tree
KNN_K = 5            # kNN baseline (stand-in for ORCA, which uses k = 5)
LOF_K = 10           # LOF setting from the paper
LOF_MAX_ROWS = 50000  # the paper has no LOF result on bigger datasets

DATA_DIR = "data"
URL = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
PYOD_URL = "https://raw.githubusercontent.com/yzhao062/pyod/master/examples/data/"  # Arrhythmia
FILES = {
    "Http": "16_http.npz", "ForestCover": "10_cover.npz",
    "Smtp": "34_smtp.npz", "Shuttle": "32_shuttle.npz",
    "Mammography": "23_mammography.npz", "Annthyroid": "2_annthyroid.npz",
    "Satellite": "30_satellite.npz", "Pima": "29_Pima.npz",
    "Breastw": "4_breastw.npz", "Arrhythmia": "arrhythmia.mat",
    "Ionosphere": "18_Ionosphere.npz",
}
# AUC in Table 3 of the paper: (iForest, ORCA, LOF); None = not reported
PAPER = {
    "Http": (1.00, 0.36, None), "ForestCover": (0.88, 0.83, None),
    "Smtp": (0.88, 0.80, None),
    "Shuttle": (1.00, 0.60, 0.55), "Mammography": (0.86, 0.77, 0.67),
    "Annthyroid": (0.82, 0.68, 0.72), "Satellite": (0.71, 0.65, 0.52),
    "Pima": (0.67, 0.71, 0.49), "Breastw": (0.99, 0.98, 0.37),
    "Arrhythmia": (0.80, 0.78, 0.73), "Ionosphere": (0.85, 0.92, 0.89),
}


def load_dataset(name):
    """Return (X, y), y = 1 for anomalies. Downloads the file on first use."""
    path = os.path.join(DATA_DIR, FILES[name])
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"  downloading {FILES[name]} ...")
        base = PYOD_URL if path.endswith(".mat") else URL
        urllib.request.urlretrieve(base + FILES[name], path)
    if path.endswith(".mat"):                       # Arrhythmia comes as a MATLAB file
        from scipy.io import loadmat
        data = loadmat(path)
        return data["X"].astype(float), data["y"].ravel().astype(int)
    data = np.load(path)
    return data["X"].astype(float), data["y"].astype(int)


def iforest_auc(X, y, seed):
    """THE ALGORITHM: scikit-learn's Isolation Forest, one run."""
    model = IsolationForest(n_estimators=N_TREES,
                            max_samples=min(SUBSAMPLE, len(X)),
                            random_state=seed)
    model.fit(X)
    # score_samples is HIGH for normal points, so negate it to get the paper's
    # anomaly score (high = anomaly). AUC needs the anomalies to score highest.
    anomaly_score = -model.score_samples(X)
    return roc_auc_score(y, anomaly_score)


def knn_auc(X, y):
    """Baseline: distance to the 5th nearest neighbour (stand-in for ORCA)."""
    dist, _ = NearestNeighbors(n_neighbors=KNN_K + 1).fit(X).kneighbors(X)
    return roc_auc_score(y, dist[:, -1])


def lof_auc(X, y):
    """Baseline: Local Outlier Factor with k = 10."""
    lof = LocalOutlierFactor(n_neighbors=LOF_K).fit(X)
    return roc_auc_score(y, -lof.negative_outlier_factor_)


def fmt(v):
    return "  n/a " if v is None else f"{v:6.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=list(FILES))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", default="results.csv", help="CSV file to write")
    args = parser.parse_args()

    print(f"IsolationForest: n_estimators={N_TREES}, max_samples={SUBSAMPLE}, "
          f"{args.runs} runs per dataset\n")
    print(f"{'dataset':12s} {'rows':>7s} {'dims':>4s} | {'iForest AUC':^30s} | "
          f"{'ORCA/kNN':^13s} | {'LOF':^13s}")
    print(f"{'':12s} {'':>7s} {'':>4s} | {'paper':>6s} {'ours':>13s} {'time':>7s} | "
          f"{'paper':>6s} {'ours':>6s} | {'paper':>6s} {'ours':>6s}")
    rows = []                                 # one dict per dataset, saved to CSV below
    for name in args.datasets:
        X, y = load_dataset(name)
        aucs, secs = [], []
        for run in range(args.runs):          # iForest is random: repeat it
            start = time.perf_counter()
            aucs.append(iforest_auc(X, y, seed=run))
            secs.append(time.perf_counter() - start)
        knn = knn_auc(X, y)
        lof = lof_auc(X, y) if len(X) <= LOF_MAX_ROWS else None
        p_if, p_orca, p_lof = PAPER[name]
        print(f"{name:12s} {len(X):7d} {X.shape[1]:4d} | {fmt(p_if)} "
              f"{np.mean(aucs):6.3f}±{np.std(aucs):.3f} {np.mean(secs):6.1f}s | "
              f"{fmt(p_orca)} {fmt(knn)} | {fmt(p_lof)} {fmt(lof)}", flush=True)
        rows.append({
            "dataset": name, "rows": len(X), "dims": X.shape[1],
            "paper_iforest_auc": p_if,
            "our_iforest_auc_mean": round(float(np.mean(aucs)), 4),
            "our_iforest_auc_std": round(float(np.std(aucs)), 4),
            "difference_ours_minus_paper": round(float(np.mean(aucs)) - p_if, 4),
            "our_iforest_time_s": round(float(np.mean(secs)), 2),
            "paper_orca_auc": p_orca,
            "our_knn_auc": round(float(knn), 4),
            "paper_lof_auc": p_lof,
            "our_lof_auc": None if lof is None else round(float(lof), 4),
        })

    with open(args.out, "w", newline="") as f:      # empty cell = not available
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
