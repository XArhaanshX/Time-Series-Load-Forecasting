# Reproducibility Configuration Log

This document records the exact software environment and hyperparameter settings used in the Haryana Short-Term Load Forecasting (STLF) research pipeline to ensure reproducibility of all experimental results.

## Software Environment

| Component | Version |
| :--- | :--- |
| **Python Interpeter** | 3.14.3 |
| **Operating System** | Windows (x64) |
| **pandas** | 2.3.3 |
| **numpy** | 2.4.2 |
| **scikit-learn** | 1.8.0 |
| **scipy** | 1.17.1 |
| **statsmodels** | 0.14.6 |
| **pyarrow** | 23.0.1 |
| **openpyxl** | 3.1.5 |
| **joblib** | 1.5.3 |
| **tqdm** | 4.67.3 |
| **xgboost** | 3.2.0 |
| **lightgbm** | 4.6.0 |
| **catboost** | 1.2.10 |

## Data Ingestion Parameters

| Parameter | Configuration Value |
| :--- | :--- |
| **Raw Data Schema** | Microsoft Excel (.xlsx) |
| **Metadata: Date Row** | 2 |
| **Standard: Header Row** | 6 |
| **Standard: Data Start Row** | 7 (Index 6) |
| **Observations per File** | 96 |
| **Sampling Frequency** | 15 Minutes |
| **Target Variable** | Haryana State Load (MW) |

## Preprocessing & Analysis Hyperparameters

| Stage | Parameter | Value |
| :--- | :--- | :--- |
| **Outlier Detection** | Winsorization Threshold (Z-Score) | 4.0 |
| **Imputation** | Forward Fill Limit | 4 (1 Hour) |
| **Stationarity** | ADF Test Confidence Level | 95% ($\alpha = 0.05$) |
| **Autocorrelation** | ACF Max Lags | 700 |
| **Autocorrelation** | PACF Max Lags | 200 |
| **Decomposition** | STL Seasonal Period | 96 |

## Feature Engineering Configuration

| Category | parameter | Settings |
| :--- | :--- | :--- |
| **Lags** | Seasonal Lag Offsets | [1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 672] |
| **Rolling** | Window Sizes (Lags) | [4, 96, 672] (1h, 1d, 1w) |
| **Fourier** | Harmonics ($k$) | 3 (Daily & Weekly cycles) |
| **Numerical** | Epsilon ($\epsilon$) | $1 \times 10^{-6}$ |
| **Cyclical** | Temporal Encodings | Sine/Cosine (Hour, Day-of-Week) |

## Experimental Design

| Component | Strategy |
| :--- | :--- |
| **Validation Method** | Expanding Window Walk-Forward Validation |
| **Dataset Splitting** | Chronological (No random shuffling) |
| **Test Horizon** | 1 Year (Final Year) |
| **Supervised Format** | Sliding Window Transformation |
| **Dataset Version** | v1_canonical_sliding_window |
| **Dataset Shape** | 142,463 samples, 3,649 features |
| **Random Seed** | 42 |
| **Total Models** | 10 (2 Baselines, 8 ML Models) |

## Deep Learning Dataset (v2)

| Component | Strategy / Value |
| :--- | :--- |
| **Dataset Version** | v2_sequence_dataset |
| **Input Format** | 3D Tensors (Samples, TimeSteps, Features) |
| **Sequence Length** | 96 (24 Hours) |
| **Feature Count** | 47 (Dynamically detected) |
| **Tensor Precision** | float32 |
| **Scaling** | StandardScaler (Features Only, Fitted on Train) |
| **Target Scale** | Original MW Magnitude (Unscaled) |
| **Train Samples** | 107,423 |
| **Test Samples** | 34,944 |
| **Split Logic** | Independent windowing per partition (96-step lag) |

## Deep Learning Dataset (v3)

| Component | Strategy / Value |
| :--- | :--- |
| **Dataset Version** | v3_sequence_dataset |
| **Input Format** | 3D Tensors (Samples, TimeSteps, Features) |
| **Sequence Length** | 192 (48 Hours) |
| **Feature Count** | 45 (Sorted, Filtered via VarianceThreshold) |
| **Tensor Precision** | float32 |
| **Sanitization** | Bidirectional Fill (ffill + bfill) |
| **Feature Scaling** | StandardScaler (Fitted on Train) |
| **Target Scaling** | StandardScaler (Fitted on Train) |
| **Train Sequences** | 107,327 |
| **Test Sequences** | 34,848 |
| **Split Logic** | Independent windowing per partition (192-step warmup) |
