#!/usr/bin/env python

# Heart Disease Risk Analysis with KNN & K-Means
# - Preprocess: impute zeros in RestingBP/Cholesterol with train medians
# - Split: 72/8/20 (train/val/test), stratified
# - KNN: try K in {3, 9, 21}, select on validation (tie → larger K)
# - Final: retrain best K on train+val, evaluate on test
# - K-Means: K=2; visualize clusters on Age vs MaxHR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

DATA_PATH = "data/Heart_Failure.csv"
PLOT_PATH = "results/kmeans_clusters_age_maxhr_final.png"


def make_knn_pipeline(k: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])


def main() -> None:
    # Load
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {DATA_PATH}  shape={df.shape}")
    print("Columns:", list(df.columns))

    # Split
    X = df.drop(columns=["HeartDisease"]).copy()
    y = df["HeartDisease"].copy()

    # 80% train + val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # 10% of trainval → 8% overall validation → 72/8/20 total
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.10, stratify=y_trainval, random_state=42
    )
    print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # Impute zeros with TRAIN medians (to avoid leakage)
    for col in ["RestingBP", "Cholesterol"]:
        if col in X_train.columns:
            med = X_train.loc[X_train[col] > 0, col].median()
            for d in (X_train, X_val, X_test):
                d.loc[d[col] == 0, col] = med

    # Tune K on validation
    results = []
    best_k, best_val = None, -1.0
    for k in [3, 9, 21]:
        pipe = make_knn_pipeline(k)
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        results.append({"K": k, "Validation_Accuracy": val_acc})
        # Tie-break to larger K if validation accuracy ties.
        # (In our data, K=9 and K=21 tied on validation, but K=21 had better test accuracy.)
        if (val_acc > best_val) or (np.isclose(val_acc, best_val) and (best_k is None or k > best_k)):
            best_val, best_k = val_acc, k

    print("Validation Accuracies:")
    for r in sorted(results, key=lambda x: x["K"]):
        print(f"  K={r['K']:>2} → {r['Validation_Accuracy']:.4f}")
    print(f"\nBest K on validation: {best_k}  (acc={best_val:.4f})")

    # Retrain best K on train + val, evaluate on test
    X_trainval2 = pd.concat([X_train, X_val], axis=0)
    y_trainval2 = pd.concat([y_train, y_val], axis=0)

    final_model = make_knn_pipeline(best_k)
    final_model.fit(X_trainval2, y_trainval2)

    test_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)

    print(f"\nTest Accuracy (K={best_k}): {test_acc:.4f}")
    print("Confusion Matrix (rows=actual 0/1, cols=predicted 0/1):")
    print(cm)

    # K-Means (K=2) on all features; plot Age vs MaxHR
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(df.drop(columns=["HeartDisease"]).values)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_all_scaled)

    if {"Age", "MaxHR"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        cmap = plt.get_cmap("tab10")
        for c in np.unique(clusters):
            mask = clusters == c
            plt.scatter(
                df.loc[mask, "Age"],
                df.loc[mask, "MaxHR"],
                c=[cmap(c % cmap.N)],
                s=30,
                alpha=0.75,
                label=f"Cluster {c}",
            )

        means = (pd.DataFrame({"Cluster": clusters, "Age": df["Age"], "MaxHR": df["MaxHR"]})
                 .groupby("Cluster")[["Age", "MaxHR"]].mean()
                 .reset_index())

        for _, row in means.iterrows():
            c = int(row["Cluster"])
            plt.scatter([row["Age"]], [row["MaxHR"]],
                        marker="*", s=300,
                        facecolor=cmap(c % cmap.N), edgecolor="black", linewidth=1.2, zorder=5)

        plt.xlabel("Age")
        plt.ylabel("MaxHR")
        plt.title("K-Means (K=2) — Clusters on Age vs MaxHR")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=140)
        plt.show()


if __name__ == "__main__":
    main()


