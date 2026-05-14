# Water Quality Anomaly Detection Using Unsupervised Machine Learning with Heterogeneous Ensemble Voting

An end-to-end framework for detecting anomalies in river water quality sensor data using three unsupervised algorithms (LOF, One-Class SVM, Isolation Forest), Causal Z-Score feature engineering, and heterogeneous voting-based ensemble. Developed as an undergraduate thesis at the Department of Information Technology, Universitas Gadjah Mada.

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Results Summary](#results-summary)
- [Model Artifacts](#model-artifacts)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

Continuous water quality monitoring through IoT sensor networks generates large volumes of unlabeled time series data that are susceptible to both sensor malfunctions and real contamination events. Traditional static thresholding methods fail to detect **contextual anomalies** — values that appear normal globally but deviate from their seasonal context (e.g., a summer-like temperature occurring in winter).

This project addresses three challenges:

1. **No ground truth labels** — The dataset has no expert-verified anomaly labels, requiring unsupervised approaches and synthetic anomaly injection for objective evaluation.
2. **Contextual anomalies are invisible to raw features** — Standard features cannot distinguish between a normal 20°C reading in summer and an anomalous 20°C reading in winter.
3. **Single-model blind spots** — Each unsupervised algorithm has inherent weaknesses; combining them through ensemble voting can improve robustness.

The framework introduces **Causal Z-Score** (a 30-day causal baseline deviation metric) as a feature engineering approach that provably outperforms both raw features and the commonly used rolling statistics, validated through a rigorous ablation study.

---

## Key Findings

**1. Feature engineering matters more than model complexity.**
Causal Z-Score consistently improved F1-Score across all three models compared to raw features (LOF: 0.747 → 0.784, OCSVM: 0.604 → 0.627, iForest: 0.586 → 0.672). Meanwhile, the popular 7-day rolling statistics approach *degraded* performance drastically (LOF dropped to 0.454) due to signal smearing effects.

**2. LOF is the best single model for this task.**
LOF (F1=0.784) outperformed both OCSVM (0.627) and iForest (0.672). Its density-based inductive bias is well-suited for detecting contextual anomalies that manifest as low-density points in the Z-Score feature space.

**3. Ensemble voting does not always beat the best single model.**
Contrary to the common assumption that "more models = better results," the heterogeneous ensemble did not surpass standalone LOF in F1-Score. This occurred because the performance disparity between models violated the *comparable competence* requirement of Condorcet's Jury Theorem. However, Vote OR achieved the highest recall (0.77 vs LOF's 0.73), which is valuable for early warning systems.

**4. Contextual anomalies are fundamentally harder to detect.**
All three models showed dramatically lower F1-Scores on contextual anomalies (LOF: 0.556) compared to global anomalies (LOF: 0.870). 100% of consensus false negatives (missed by all models) were contextual anomalies.

---

## Dataset

| Property | Value |
|---|---|
| Source | United States Geological Survey (USGS) |
| Location | Clackamas River, Oregon, USA |
| Period | January 1, 2014 – December 31, 2023 (10 years) |
| Granularity | Daily |
| Total Days | 3,652 (after reindexing) |
| Parameters | Temperature (°C), Dissolved Oxygen (mg/L), pH, Turbidity (NTU) |
| License | Public Domain |

The raw CSV contains 6,209 rows and 34 columns. After filtering to the four target parameters, removing malformed datetime entries, and reindexing to a complete daily range, the dataset comprises 3,652 daily records.

---

## Methodology

### Pipeline Overview

```
Raw CSV → EDA → Cleaning/Imputation → Time-Based Split → Synthetic Anomaly Injection
    → Feature Engineering (Causal Z-Score) → RobustScaler → Hyperparameter Tuning
    → Model Training → Evaluation → Ablation Study → Ensemble Voting → Save Model
```

### Data Splitting (Time-Based)

| Split | Period | Days | Proportion |
|---|---|---|---|
| Training | 2014–2020 | 2,557 | 70% |
| Validation | 2021 | 365 | 10% |
| Test | 2022–2023 | 730 | 20% |

Time-based splitting ensures no data leakage from future to past, reflecting real-world deployment conditions.

### Synthetic Anomaly Injection

Since no ground truth labels exist, synthetic anomalies are injected into the validation and test sets with a balanced 50/50 composition:

| Type | Subtype | Description |
|---|---|---|
| Global (univariate) | Turbidity Spike | Turbidity set to 3x the 99th percentile of training data |
| Global (multivariate) | Thermal Pollution | Temperature +6°C and DO -3 mg/L simultaneously |
| Contextual (univariate) | Unusual Seasonal Temp | Temperature set to monthly_mean + 3×monthly_std |
| Contextual (multivariate) | Season Swap | Temperature and DO swapped with values from the opposite month (±6 months) |

Injection counts: 72 anomalies in validation set (seed=42), 144 in test set (seed=143).

### Feature Engineering: Causal Z-Score

The core innovation of this project. For each parameter, the Causal Z-Score is computed as:

```
Z_causal(x_t) = (x_t - mean(x_{t-31}...x_{t-1})) / (std(x_{t-31}...x_{t-1}) + epsilon)
```

Key properties:
- **Causal**: Uses `.shift(1)` so that the current day does not contaminate its own baseline, preventing data leakage.
- **30-day window**: Aligns with monthly seasonal variation in river water quality.
- **Contextual sensitivity**: A 20°C reading in winter produces a high Z-Score (baseline ~5°C), while the same reading in summer produces a low Z-Score (baseline ~20°C).

### Models

| Model | Inductive Bias | Complexity | Best Params (Validation) |
|---|---|---|---|
| Local Outlier Factor (LOF) | Local density comparison | O(n²) | n_neighbors=15, contamination=0.03 |
| One-Class SVM (OCSVM) | Global margin boundary | O(n²)–O(n³) | nu=0.01, gamma='auto' |
| Isolation Forest (iForest) | Random tree isolation | O(n) | n_estimators=100, contamination=0.10 |

### Ensemble Voting Strategies

| Strategy | Rule | Use Case |
|---|---|---|
| Vote OR | Anomaly if ≥1 model flags | Early warning (maximize recall) |
| Vote Majority | Anomaly if ≥2 models flag | Balanced applications |
| Vote AND | Anomaly if all 3 models flag | Public alerts (maximize precision) |

---

## Project Structure

```
.
├── data/
│   └── clackamas_river.csv          # Raw USGS dataset
├── models/
│   └── anomaly_detector.pkl         # Saved model artifacts (joblib)
├── main.ipynb                       # Main experiment notebook (72 cells)
└── README.md
```

### Notebook Structure (main.ipynb)

| Step | Description | Cells |
|---|---|---|
| 0 | Problem Statement | 0–4 |
| 1 | Import Libraries | 5–6 |
| 2 | Data Loading and Inspection | 7–11 |
| 3 | Exploratory Data Analysis | 12–22 |
| 4 | Data Cleaning and Imputation | 23–25 |
| 5 | Time-Based Splitting | 26–27 |
| 6 | Synthetic Anomaly Injection | 28–31 |
| 7 | Feature Engineering (Rolling + Z-Score) | 32–36 |
| 8 | Standardization (RobustScaler) | 37–38 |
| 9 | Feature Set Definitions for Ablation | 39–40 |
| 10 | Hyperparameter Tuning (Grid Search) | 41–50 |
| 11 | Final Model Evaluation on Test Set | 51–58 |
| 11.D | Performance Analysis by Anomaly Type | — |
| 11.E | Error Analysis (Consensus False Negatives) | — |
| 12 | Ablation Study | 59–62 |
| 12.B | Ablation by Anomaly Type | — |
| 13 | Heterogeneous Ensemble Voting | 63–68 |
| 14 | Conclusions and Recommendations | 69 |
| 15 | Save Model | 70–71 |

---

## Installation and Usage

### Requirements

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- matplotlib
- seaborn
- joblib

### Setup

```bash
# Clone the repository
git clone https://github.com/aDJi2003/water-quality-anomaly-detection.git
cd water-quality-anomaly-detection

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run the notebook
jupyter notebook main.ipynb
```

### Using the Saved Model for Inference

```python
import joblib
import pandas as pd
import numpy as np

# Load artifacts
artifacts = joblib.load('models/anomaly_detector.pkl')
lof = artifacts['lof']
svm = artifacts['ocsvm']
iso = artifacts['iforest']
scaler = artifacts['scaler']
feature_cols = artifacts['feature_columns']

def detect_anomalies(df_new):
    """
    Detect anomalies in new daily water quality data.
    Requires at least 31 days of history for Causal Z-Score computation.
    
    Parameters
    ----------
    df_new : pd.DataFrame
        DataFrame with DatetimeIndex and columns:
        Temperature, Dissolved Oxygen, pH, Turbidity
    
    Returns
    -------
    pd.DataFrame
        Original data with anomaly predictions and voting results.
    """
    df = df_new.copy()
    
    # Feature engineering: Causal Z-Score (30-day baseline)
    for col in ['Temperature', 'Dissolved Oxygen', 'pH', 'Turbidity']:
        shifted = df[col].shift(1)
        roll_mean = shifted.rolling(30, min_periods=10).mean()
        roll_std = shifted.rolling(30, min_periods=10).std()
        df[f'{col}_ZScore'] = (df[col] - roll_mean) / (roll_std + 1e-6)
    
    # Cyclical month encoding
    df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Drop rows without enough history
    df = df.dropna(subset=feature_cols)
    
    # Scale features
    cols_to_scale = [c for c in feature_cols if c not in ['Month_Sin', 'Month_Cos']]
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # Predict with each model
    X = df_scaled[feature_cols].values
    df['pred_lof'] = [1 if x == -1 else 0 for x in lof.predict(X)]
    df['pred_svm'] = [1 if x == -1 else 0 for x in svm.predict(X)]
    df['pred_iso'] = [1 if x == -1 else 0 for x in iso.predict(X)]
    
    # Ensemble voting
    votes = df['pred_lof'] + df['pred_svm'] + df['pred_iso']
    df['vote_or'] = (votes >= 1).astype(int)
    df['vote_majority'] = (votes >= 2).astype(int)
    df['vote_and'] = (votes >= 3).astype(int)
    
    return df
```

---

## Results Summary

### Individual Model Performance (Test Set, Raw + ZScore Features)

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| **LOF (Tuned)** | **0.85** | **0.73** | **0.784** | **0.92** |
| iForest (Tuned) | 0.76 | 0.60 | 0.672 | 0.88 |
| OCSVM (Tuned) | 0.82 | 0.51 | 0.627 | 0.88 |

### Ensemble Voting Performance (Test Set)

| Strategy | Precision | Recall | F1-Score |
|---|---|---|---|
| Vote OR | 0.74 | 0.77 | 0.758 |
| Vote Majority | 0.83 | 0.60 | 0.694 |
| Vote AND | 0.91 | 0.47 | 0.621 |

### Ablation Study: Effect of Feature Engineering on F1-Score

| Feature Scenario | # Features | LOF | OCSVM | iForest |
|---|---|---|---|---|
| Raw Only | 4 | 0.747 | 0.604 | 0.586 |
| Raw + Rolling 7D | 18 | 0.454 | 0.406 | 0.424 |
| **Raw + ZScore 30D** | **10** | **0.784** | **0.627** | **0.672** |

### Performance by Anomaly Type (Best Model: LOF with ZScore)

| Anomaly Type | Recall | F1-Score |
|---|---|---|
| Global | 0.972 | 0.870 |
| Contextual | 0.486 | 0.556 |

### Error Analysis

- 33 out of 144 anomalies were missed by all three models (consensus false negatives)
- 100% of consensus false negatives were contextual anomalies
- Most occurred during seasonal transition months (Feb, Mar, Oct, Nov) where the difference between normal and swapped seasonal patterns is minimal

---

## Model Artifacts

The saved model file (`models/anomaly_detector.pkl`) contains:

| Key | Description |
|---|---|
| `lof` | Trained LOF model (n_neighbors=15, contamination=0.03) |
| `ocsvm` | Trained OCSVM model (nu=0.01, gamma='auto') |
| `iforest` | Trained Isolation Forest model (n_estimators=100, contamination=0.10) |
| `scaler` | Fitted RobustScaler (fit on training set only) |
| `feature_columns` | List of 10 feature names used by models |
| `base_features` | List of 4 raw parameter names |
| `cols_to_scale` | Columns that undergo scaling |
| `passthrough_cols` | Columns excluded from scaling (Month_Sin, Month_Cos) |
| `best_params` | Dictionary of best hyperparameters per model |

---

## Limitations

- **Single river validation**: All experiments use Clackamas River data only. Generalization to rivers with different hydrological characteristics (e.g., tropical rivers without cold-season variation) has not been validated.
- **Synthetic anomalies**: Evaluation relies on injected anomalies, which may not fully represent the complexity of real-world contamination events.
- **Daily granularity**: Sub-daily anomalies (hourly or per-minute spikes) cannot be detected. Adaptation to higher-frequency data requires methodology changes.
- **No deep learning comparison**: LSTM Autoencoders and Transformer-based detectors were not evaluated, leaving open the question of whether deep temporal models could better capture long-range contextual patterns.
- **Voting-only ensemble**: Score-level fusion, stacking, and adaptive weighted voting were not explored.

---

## Future Work

1. **Deep learning temporal models** — LSTM Autoencoders or Transformer-based anomaly detectors for capturing long-range temporal dependencies beyond the 30-day Z-Score window.
2. **Active learning** — Selectively querying domain experts to label ambiguous cases near the decision threshold, progressively improving model accuracy with minimal labeling effort.
3. **Multi-river validation** — Testing on rivers with diverse hydrological and climatic characteristics to assess generalizability of the Causal Z-Score approach.
4. **Window-based detection** — Evaluating sequences of consecutive days rather than individual points to detect collective anomalies (e.g., gradual algal blooms).
5. **Adaptive seasonal thresholds** — Using extreme value theory to dynamically adjust contamination rates per season.
6. **Selective ensemble** — Dynamically choosing the best detector per data region to address the performance disparity issue found in this study.
7. **Real-time streaming deployment** — Integrating the model into a streaming pipeline with incremental Z-Score updates and online missing value handling.

---

## References

Key references used in this project:

1. Breunig, M.M., Kriegel, H.-P., Ng, R.T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *ACM SIGMOD Record*, 29(2), 93–104.
2. Scholkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., & Williamson, R.C. (2001). Estimating the Support of a High-Dimensional Distribution. *Neural Computation*, 13(7), 1443–1471.
3. Liu, F.T., Ting, K.M., & Zhou, Z.-H. (2008). Isolation Forest. *Proc. 8th IEEE ICDM*, 413–422.
4. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1–58.
5. Zimek, A., Campello, R.J.G.B., & Sander, J. (2014). Ensembles for Unsupervised Outlier Detection. *ACM SIGKDD Explorations*, 15(1), 11–22.
6. Leigh, C., et al. (2019). A Framework for Automated Anomaly Detection in High Frequency Water-Quality Data. *Science of The Total Environment*, 664, 885–898.
7. Kholkin, V., et al. (2023). Anomaly Detection in Biological Early Warning Systems Using Unsupervised Machine Learning. *Sensors*, 23(5), 2687.
8. Thomas, R. & Judith, J.E. (2020). Voting-Based Ensemble of Unsupervised Outlier Detectors. *LNEE*, 656, 501–511.

For the complete list of 30 references, see the thesis document.

---

## License

This project is developed for academic purposes as an undergraduate thesis. The USGS dataset used is in the public domain. The code is available for educational and research use.