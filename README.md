# Short-Term Load Forecasting for Haryana Power Grid

This research project implements a full ML pipeline for predicting the 15-minute granularity power load of the Haryana grid.

## Project Structure

```
load_forecasting_project/
├── data/
│   ├── raw/                     # Original XLSX files
│   ├── processed/               # Cleaned datasets
│   └── splits/                  # Training/Val/Test splits
├── src/
│   ├── data_processing/         # Ingestion and Preprocessing
│   ├── feature_engineering/     # Rolling, Lags, Calendar features
│   ├── models/                  # ML/DL models (LSTM, Transformer, etc.)
│   ├── training/                # Training loops
│   ├── evaluation/              # Metrics and Plots
│   └── utils/                   # Config and Logger
├── models/                      # Saved models and scalers
└── results/                     # Metric reports and plots
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Stage 1 & 2 (Ingestion + Preprocessing):
   ```bash
   python src/data_processing/ingest_and_preprocess.py
   ```

## Preprocessing Pipeline

1. **Ingestion**: Dynamic discovery of headers and dates in Excel files.
2. **Frequency Enforcement**: Ensuring strict 15-minute intervals using `asfreq`.
3. **Imputation**: 
   - Forward fill (small gaps)
   - Seasonal ($t-96$ steps)
   - Rolling mean
4. **Outlier Handling**: IQR detection replaced with rolling median.
5. **Normalization**: MinMax scaling (scaler saved to `models/scalers/`).

## Execution

You can run the full pipeline using:
```bash
python run_pipeline.py
```
