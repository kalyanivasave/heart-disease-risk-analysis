# Heart Disease Risk Analysis

A compact machine-learning project that demonstrates supervised classification (K-Nearest Neighbors) and unsupervised clustering (K-Means) on a heart disease dataset. The analysis predicts heart-disease presence and explores relationships between features such as Age and Maximum Heart Rate (MaxHR).

## Project structure

```
heart-disease-risk-analysis/
├─ data/
│  └─ Heart_Failure.csv
├─ notebooks/
│  └─ HW1_Code.ipynb
├─ src/
│  └─ HW1_Code.py
├─ results/
│  ├─ kmeans_clusters_age_maxhr_final.png
│  └─ HW1_Result.pdf
├─ requirements.txt
└─ run.sh
```

## Quick start

1. Create and activate a Python virtual environment (macOS / Linux):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the analysis:

- With the helper script:

```bash
chmod +x run.sh
./run.sh
```

- Or directly:

```bash
python src/HW1_Code.py
```

4. To explore interactively, open the notebook:

```bash
jupyter lab notebooks/HW1_Code.ipynb
```

## Usage (example)

Run the script from the repository root. Representative console output:

```
Loaded: data/Heart_Failure.csv  shape=(300, 15)
Train=216  Val=24  Test=60
Validation Accuracies: K=3 -> 0.6351, K=9 -> 0.6486, K=21 -> 0.6486
Best K on validation: 21 (acc=0.6486)
Test Accuracy (K=21): 0.7283
Confusion Matrix:
[[52 30]
 [20 82]]
Saved plot: results/kmeans_clusters_age_maxhr_final.png
```

(Exact numbers may vary depending on dataset and environment.)

## Implementation summary

- Data: `data/Heart_Failure.csv` (features include Age, RestingBP, Cholesterol, MaxHR, etc.)
- Preprocessing: imputes zero values in `RestingBP` and `Cholesterol` using training-set medians (avoids leakage)
- Modeling: KNN evaluated on k in {3, 9, 21}, selected by validation accuracy; final model retrained on train+val and evaluated on test
- Clustering: K-Means (k=2) used to explore structure; visualization saved to `results/kmeans_clusters_age_maxhr_final.png`

## Notes

- `src/HW1_Code.py` is the primary script; update `DATA_PATH` inside the script if your working directory differs.
- The notebook (`notebooks/HW1_Code.ipynb`) mirrors the script and is useful for exploration and plotting.
- For reproducible installs, consider pinning the dependency versions in `requirements.txt`.
