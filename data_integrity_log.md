# Data Integrity and Leakage Prevention Log

This document outlines the rigorous verification procedures and architectural safeguards implemented to ensure the integrity of the Haryana load forecasting dataset and to prevent any form of temporal data leakage.

## 1. Temporal Data Leakage Safeguards

In time-series forecasting, data leakage occurs if information from the future (relative to the prediction target $t$) is inadvertently used to compute features for the model. The following safeguards are enforced:

### Shift-Before-Compute Architecture
All rolling statistics, differences, and lag features are computed using shifted versions of the target series. 
*   **Rolling Mean**: $\mu_{t, N} = \frac{1}{N} \sum_{i=1}^{N} L_{t-i}$. 
*   **Momentum**: $\Delta L_{t, 1} = L_{t-1} - L_{t-2}$.
*   **Implementation Rule**: Every feature engineering function in `src/feature_engineering.py` applies a `shift(1)` operation to the load series before performing any aggregation or differencing.

### Sliding Window Integrity
The sliding window transformation in `src/dataset_construction.py` is strictly causal. For a target at time $t$, only the feature vectors from $[t-o, \dots, t-1]$ are included in the feature matrix.

### Automated Verification
The pipeline executes a `validate_causality` check which confirms that for every observation $i$ in the supervised dataset:
$$Timestamp(Features_{i, \text{latest}}) < Timestamp(Target_{i})$$

## 2. Ingestion Integrity Verification

The ingestion pipeline (`src/ingestion.py`) incorporates strict boundary checks to prevent contamination from non-state load data:

*   **Sub-Table Truncation**: District-level load tables (often located below the 96th row) are excluded by explicitly limiting the `pd.read_excel` call to `nrows=96` starting from the discovered Haryana state header.
*   **Metadata Validation**: The `REPORT FOR THE DAY` date is extracted from Row 2 and compared against the file name's encoded date to ensure chronological alignment.
*   **Row-Level Count Verification**: Any Excel file failing to provide exactly 96 rows of parseable 15-minute data triggers a warning/rejection to ensure the daily sum is not compromised by missing data.

## 3. Preprocessing Integrity

*   **Frequency Continuity**: The `asfreq('15min')` operation identifies all missing intervals. Gaps are primarily resolved through forward-filling (limit=4) to ensure model stability while recording the percentage of synthetic data injected.
*   **Duplicate Detection**: Duplicate timestamps (a common artifact in manual log entries) are dropped, preserving only the first entry to maintain a strictly monotone increasing time index.
*   **NaN Audit**: A final check is performed on the feature matrix prior to model training; any row containing a `NaN` value is dropped, ensuring the model is not trained on incomplete windows.

## 4. Baseline Metric Correction Verification

A verification audit of the classical ML results confirmed and corrected an initial discrepancy in the baseline metrics:

*   **Correction Scope**: The metrics for the `Persistence` and `Seasonal Persistence` models were recomputed using the true target series.
*   **Methodology**: The recomputation utilized the finalized evaluation logic to ensure consistency across all benchmark models.
*   **Data Integrity Preservation**: This correction was performed as an isolated evaluation update. No modifications were made to the canonical dataset files (`.parquet`), and no retraining of machine learning models was required.
*   **Resolution**: The recomputed baseline metrics accurately reflect the short-term autocorrelation and daily periodicity of the Haryana load series, providing a robust basis for model comparison.
