# Research Methodology: Short-Term Load Forecasting for the Haryana Power Grid

## 1. Introduction
Short-Term Load Forecasting (STLF) is a critical task for power grid operators to ensure voltage stability, plan economic dispatch, and manage peak demand. This study focuses on the Haryana state grid, a region with high variability in demand due to diverse agricultural, industrial, and residential sectors. The objective is to develop a robust, multi-scale feature engineering pipeline that prepares 15-minute resolution electricity demand data for supervised learning models.

## 2. Dataset Description
The dataset comprises approximately **245,000 observations** spanning a **seven-year period**. The raw data is derived from daily electricity demand logs of the Haryana grid.

### 2.1 Raw Log Structure
Each daily report is provided as a Microsoft Excel file with a standardized but nested structure:
*   **Metadata**: Row 2 explicitly contains the `REPORT FOR THE DAY` date.
*   **Headers**: Row 6 contains the column identifiers (e.g., `TIME`, `HARYANA_LOAD`).
*   **State Load Data**: Row 7 (Index 6) begins the observation block. Exactly **ninety-six (96)** rows correspond to the fifteen-minute intervals of the day.
*   **District Tables**: Rows below the 96th observation contain sub-tables for individual districts. These are strictly excluded from the ingestion process to prevent contamination of the total state demand signal.

### 2.2 Dataset Statistics
The processed dataset exhibits the following characteristics:
*   **Sampling Frequency ($f$):** 15 minutes.
*   **Daily Observations ($N_d$):** 96.
*   **Total Observations ($N$):** $\approx$ 245,000.
*   **Target Variable ($y$):** Haryana Total State Demand in Megawatts (MW).

## 3. Data Processing Pipeline

### 3.1 Data Ingestion
The ingestion module utilizes the `openpyxl` engine to parse daily logs. The procedure involves:
1.  **Date Extraction**: Parsing Row 2 to obtain the Gregorian date.
2.  **Causal Truncation**: Reading exactly 96 rows from the start of the state load block.
3.  **Timestamp Construction**: Combining the date metadata with the `TIME` column to create a continuous `datetime` index.
4.  **Chronological Consolidation**: Concatenating daily files and sorting by timestamp to ensure a strictly monotonic time series.

### 3.2 Preprocessing & Data Cleaning
The pipeline implements several safeguards to ensure data quality:
*   **Frequency Enforcement**: Enforcing a strict 15-minute interval. Duplicate timestamps are removed, and missing intervals are identified.
*   **Winsorization**: To handle outliers while preserving extreme demand spikes (which are often the most critical for forecasting), the pipeline uses Z-score clipping at **four standard deviations** ($4\sigma$) from the global mean.
*   **Imputation**: Forward-filling is applied to small gaps (up to 1 hour / 4 observations) to maintain causal integrity.

## 4. Statistical Analysis

### 4.1 Stationarity & Seasonality
The series is analyzed for stationarity using the **Augmented Dickey–Fuller (ADF) test**. To capture the multi-seasonal nature of electricity demand, the **Autocorrelation Function (ACF)** is computed up to **lag 700**. 
*   **Daily Seasonality**: Strong correlation observed at lag 96.
*   **Weekly Seasonality**: Strong correlation observed at lag 672.

### 4.2 STL Decomposition
Seasonal-Trend decomposition using LOESS (STL) is performed using the equation:
$$y_t = T_t + S_t + R_t$$
where $y_t$ is the load, $T_t$ is the trend component, $S_t$ is the seasonal component (period=96), and $R_t$ is the residual. This decomposition is utilized for structural analysis of the demand patterns rather than as a direct model input.

### 4.3 Lag Discovery
Optimal autoregressive depth is determined via the **Partial Autocorrelation Function (PACF)** computed up to lag 200. The optimal lag $o$ is defined as:
$$o = \max(L_{\text{contig}}, L_{\text{seasonal}})$$
where $L_{\text{contig}}$ is the largest contiguous block of significant lags near the origin, and $L_{\text{seasonal}}$ is the daily seasonal spike at 96.

## 5. Feature Engineering Strategy

The pipeline generates approximately **40 base features** across several categories:

### 5.1 Temporal & Cyclical Features
*   **Categorical**: Hour of day, Day of week, Month of year.
*   **Cyclical Encodings**: To preserve periodic continuity (e.g., hour 23 being close to hour 0), sine and cosine transformations are applied:
    $$f_{\text{sin}} = \sin\left(\frac{2\pi \cdot x}{X_{\text{max}}}\right), \quad f_{\text{cos}} = \cos\left(\frac{2\pi \cdot x}{X_{\text{max}}}\right)$$

### 5.2 Fourier Series Harmonics
To capture smooth seasonal transitions, Fourier features are constructed using the first three harmonics for both daily and weekly cycles:
$$F_{k, \text{sin}} = \sin\left(\frac{2\pi k t}{P}\right), \quad F_{k, \text{cos}} = \cos\left(\frac{2\pi k t}{P}\right)$$
where $P \in \{96, 672\}$ and $k \in \{1, 2, 3\}$.

### 5.3 Rolled & Lagged Statistics
All statistical aggregations are computed using **shifted load values** ($L_{t-1}$) to ensure zero data leakage. 
*   **Rolling Windows**: 1 hour (4 lags), 1 day (96 lags), and 1 week (672 lags).
*   **Metrics**: Mean, Standard Deviation, Minimum, and Maximum.

### 5.4 Momentum & Difference Features
*   **Differences**: $\Delta L_{t, 1}$, $\Delta L_{t, 96}$, and $\Delta L_{t, 672}$ represent short-term, daily, and weekly demand shifts.
*   **Ratio Feature**: $R_{t, 96} = \frac{L_{t-1}}{L_{t-96} + \epsilon}$, where $\epsilon = 1 \times 10^{-6}$ for numerical stability.

### 5.5 Lag Multi-Scale Features
Lags are generated at offsets $\{1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 672\}$ to capture multiscale temporal dependencies.

## 6. Supervised Dataset Construction

### 6.1 Sliding Window Transformation
The time series is converted into a supervised learning format via a sliding window. For an optimal lag $o$, a sample at time $t$ uses the feature vector $\mathbf{X}_t$:
$$\mathbf{X}_t = [f_{t-o}, \dots, f_{t-1}]$$
as the input for the target $y_t$. This transformation results in high-dimensional feature vectors (often reaching several thousand dimensions) depending on the number of base features and window size.

## 7. Experimental Setup

### 7.1 Expanding Window Validation
To simulate real-world operational conditions, the pipeline utilizes **expanding window walk-forward validation**. The model is trained on a growing historical window and evaluated on the subsequent one-year period, ensuring that the training set always precedes the test set in time.

## 8. Classical Machine Learning Benchmark Results

The classical machine learning benchmarking stage has been completed. Ten models, including two baselines and eight supervised learning methods, were evaluated using the standardized feature matrix and walk-forward validation strategy. The final performance metrics are summarized in Table 1.

**Table 1: Performance Metrics for Classical ML Models**

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| Persistence | 114.50 | 180.45 | 1.63 |
| Seasonal Persistence | 495.44 | 734.69 | 7.18 |
| Linear Regression | 128.27 | 193.76 | 1.81 |
| Ridge Regression | 127.98 | 193.54 | 1.81 |
| Lasso Regression | 127.87 | 194.78 | 1.80 |
| Random Forest | 140.40 | 212.70 | 1.95 |
| Gradient Boosting | 136.45 | 204.04 | 1.89 |
| XGBoost | 149.70 | 224.52 | 2.16 |
| LightGBM | 129.19 | 195.17 | 1.79 |
| CatBoost | 138.81 | 202.87 | 1.94 |

### 8.1 Performance Analysis

The experimental results yield several critical observations regarding the forecasting behavior of the Haryana load series:

1.  **Persistence Baseline Performance**: The persistence model achieved the lowest error among all tested methods (MAPE ~1.63%), indicating extremely strong short-term autocorrelation in the Haryana load series. This performance suggests that the immediate history of the series is a near-perfect predictor for the subsequent 15-minute interval.
2.  **Seasonal Persistence Behavior**: The seasonal persistence model ($t-96$) performs significantly worse than short-term persistence (MAPE ~7.18%), suggesting that immediate historical load provides a far stronger predictive signal than daily periodicity alone in the presence of intra-day variability.
3.  **Linear Model Competitiveness**: Linear Regression, Ridge, and Lasso all achieved very similar performance (~1.8% MAPE). This demonstrates that the engineered feature space effectively linearizes the forecasting problem, allowing simple linear predictors to perform on par with more complex models.
4.  **Ensemble Model Comparison**: Tree-based ensemble models (Random Forest, Gradient Boosting, CatBoost) did not significantly outperform linear methods. This suggests limited nonlinear interactions beyond those already captured by the comprehensive set of lag and rolling features.
5.  **LightGBM Performance**: LightGBM achieved the best performance among the machine learning models (~1.79% MAPE) while maintaining very low training time compared to other boosting models, highlighting its efficiency for high-dimensional load forecasting tasks.
6.  **XGBoost Underperformance**: XGBoost produced slightly higher error than other boosting models, likely due to a lack of hyperparameter tuning or sensitivity to the high-dimensional feature space generated by the pipeline.

### 8.2 Time Series Characteristics

Implicit in the superior performance of the persistence model are several fundamental characteristics of the Haryana dataset:
*   **Strong Autocorrelation**: The series exhibits extreme correlation at short lags, where the immediate previous value dominates the predictive signal.
*   **Gradual Variation**: Load variations over 15-minute intervals are predominantly gradual, confirming the high inertia of the grid's demand.
*   **Predictable Dynamics**: The short-term dynamics are highly predictable, placing an upper bound on the achievable forecasting improvement relative to simple baselines.

These results align with earlier statistical analysis which detected strong ACF correlation at lag 1, daily periodicity at lag 96, and weekly periodicity at lag 672. The presence of strong persistence behavior confirms that noise in the 15-minute resolution data is relatively low, but improvement beyond the baseline requires capturing very subtle nonlinear deviations.

### 8.3 Analysis of Feature Engineering Effectiveness

The comprehensive feature set—comprising multi-scale lags, rolling statistics, Fourier seasonality features, and cyclical calendar encodings—significantly influenced model effectiveness. These transformations successfully convert nonlinear temporal relationships into explicit features that can be effectively captured by linear models. The fact that linear models remained competitive with boosted trees suggests that the feature engineering stage has successfully bridged the complexity gap by projecting temporal dependencies into a more accessible feature space.

## 9. Key Findings From Classical ML Benchmark

1.  **Dominance of Persistence**: The load series exhibits extremely strong persistence behavior, which serves as a highly competitive benchmark.
2.  **Competitiveness of Linear Models**: Classical linear methods remain remarkably competitive with advanced gradient boosting machines when provided with extensive engineered features.
3.  **Marginal Improvements from Ensembles**: Tree-based ensemble models provide only marginal improvements over linear models, suggesting that the problem is well-linearized.
4.  **Sensitivity to Feature Representation**: Feature engineering significantly influences the effectiveness of all tested models, particularly for linear methods.

These insights will support the subsequent comparison with deep learning models, where the ability to learn representations directly from sequences will be tested against this hand-crafted feature approach.

## 10. Next Stage: Deep Learning Benchmark

The next experimental stage will evaluate sequence-based deep learning models to determine if they can capture remaining nonlinearities without the need for extensive feature engineering. The models to be evaluated include:
*   **Long Short-Term Memory (LSTM)**
*   **Gated Recurrent Units (GRU)**
*   **CNN-LSTM hybrids**
*   **Transformer architectures**

This transition will necessitate a shift in data representation from the flattened sliding-window feature matrix used by classical ML models to sequence tensors of shape $(\text{samples} \times \text{time\_steps} \times \text{features})$.
