# Household Power Consumption — KMeans Clustering

This project groups minute-by-minute electricity readings from a household into clusters of similar consumption behavior, using **KMeans**. The goal is to discover distinct "usage patterns" (for example, weekday-evening peaks vs. quiet overnight periods) without any labels.

Everything lives in one notebook: `Household_Power_Clustering_Improved.ipynb`.

---

## Dataset

**UCI "Individual household electric power consumption"** (`household_power_consumption.txt`), a semicolon-separated file with roughly 2 million one-minute readings.

| Column | Meaning |
|---|---|
| `Date`, `Time` | Timestamp of the reading (`dd/mm/yyyy`, `hh:mm:ss`) |
| `Global_active_power` | Household active power (kW) |
| `Global_reactive_power` | Household reactive power (kW) |
| `Voltage` | Average voltage (V) |
| `Global_intensity` | Average current (A) |
| `Sub_metering_1` | Energy use in the kitchen (Wh) |
| `Sub_metering_2` | Energy use in the laundry room (Wh) |
| `Sub_metering_3` | Energy use by water heater and air conditioner (Wh) |

Missing values are stored as `?` in the file and are read in as `NaN`.

---

## How it works (pipeline)

```
Load → Clean → Handle outliers → Engineer time features → Drop redundant features
     → Fix skew → Scale → Choose k → Fit KMeans → Evaluate → Profile → Visualize (PCA)
```

### 1. Load
The file is read with `pd.read_csv(..., sep=';', na_values=['?'])`.

### 2. Data quality
Duplicate rows are counted and dropped.

### 3. Missing values
Each of the 7 numeric columns is filled with its own column mean.

### 4. Outlier treatment
KMeans relies on Euclidean distance, so extreme values in any feature can pull centroids around. Every numeric column is **IQR-clipped**: values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` are clipped to those bounds rather than removed, so no rows are lost.

### 5. Time feature engineering
`Date` and `Time` are combined into a datetime and turned into three numeric features:
- `Hour` (0–23)
- `DayOfWeek` (0 = Monday)
- `IsWeekend` (0/1)

*When* power is used is often as informative as *how much*, so these are kept as clustering features instead of being discarded.

### 6. Multicollinearity check
A correlation matrix of the power features is plotted. For any pair with |correlation| > **0.95**, one feature is dropped automatically. In this dataset `Global_active_power` and `Global_intensity` are nearly perfectly correlated, so `Global_intensity` is removed. Otherwise the same underlying signal would be counted twice in the distance calculation.

### 7. Skew handling
Skewness is computed per feature. Any feature with |skew| > 1 gets a `log1p` transform (shifted to be non-negative first), since power data is typically right-skewed and `StandardScaler` does not fix that.

### 8. Scaling
`StandardScaler` puts all features on a mean-0, std-1 scale so no feature dominates purely because of its units.

### 9. Choosing the number of clusters (k)
KMeans is run for **k = 2 … 8**, recording:
- **Inertia / WCSS** for the elbow plot
- **Silhouette score** for each k

The default `FINAL_K` is the k with the highest silhouette score. You can override it manually after checking the elbow plot.

### 10. Final model
`KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)` is fit on the scaled data and each row receives a `Cluster` label.

### 11. Evaluation
Three metrics are reported on the fitted model:

| Metric | Better when |
|---|---|
| Silhouette score | Higher (max 1.0) |
| Davies–Bouldin index | Lower |
| Calinski–Harabasz index | Higher |

A per-sample silhouette plot (on a 20,000-row sample, for speed) shows weakly assigned points that the average score can hide.

### 12. Cluster profiling
Feature means are computed per cluster and shown as a table with cluster sizes, plus a z-scored heatmap so each cluster's distinguishing traits stand out (e.g. "high power, evening hours, weekdays"). The notebook includes an interpretation template for you to fill in with your own cluster descriptions.

### 13. PCA visualization
The scaled data is projected to 2 principal components and plotted, colored by cluster, with the explained variance shown on the axes.

---

## Getting started

### Requirements
- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run it
1. Download `household_power_consumption.txt` from the UCI Machine Learning Repository and place it next to the notebook.
2. Open the notebook:
   ```bash
   jupyter notebook Household_Power_Clustering_Improved.ipynb
   ```
3. **Fix the data path.** The notebook defines `DATA_PATH = 'household_power_consumption.txt'`, but the `read_csv` call still uses a hardcoded Windows path. Change it to `pd.read_csv(DATA_PATH, ...)`.
4. Run all cells. Results are reproducible because `random_state=42` is used throughout.

> Note: silhouette scoring on ~2M rows is slow and memory-hungry. If the k-selection loop takes too long, compute `silhouette_score` on a random sample (the per-sample plot already does this).

---

## Known limitations

- **Sparse sub-metering columns.** `Sub_metering_1` and `Sub_metering_2` are zero most of the time. With IQR clipping, if Q1 and Q3 are both 0 the upper bound is 0, which can flatten these columns to a constant. Check the post-clipping boxplots and consider skipping clipping for these columns.
- **Single household.** The data comes from one home, so clusters describe *time periods* of that home's usage, not different households.
- **Minute-level noise.** At ~2M rows, clusters may reflect short-term fluctuations more than meaningful behavior.
- **KMeans assumptions.** It favors roughly spherical, similarly sized clusters, and time features like `Hour` are treated as linear (23:00 and 00:00 look far apart).

## Ideas for improvement

- Aggregate to hourly or daily profiles before clustering.
- Encode `Hour` cyclically (sine/cosine).
- Compare KMeans with DBSCAN, Gaussian mixtures, or hierarchical clustering.
- Add the derived "unmetered consumption" feature:
  `Global_active_power * 1000 / 60 − (Sub_metering_1 + Sub_metering_2 + Sub_metering_3)`.