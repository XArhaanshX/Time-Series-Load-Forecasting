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

## 10. Deep Learning Methodology

### 10.1 Overview of Deep Learning Approach
Deep learning architectures were explored for Short-Term Load Forecasting (STLF) to evaluate their ability to autonomously extract complex temporal dependencies without the extensive manual feature engineering required by classical models. Electricity demand is inherently a non-stationary time series characterized by multi-scale seasonality (daily, weekly, and seasonal) and nonlinear responses to temporal and exogenous factors. 

Sequential modeling is critical for STLF as the state of the power grid is highly correlated with its immediate history. Architectures such as **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)** are particularly relevant due to their gating mechanisms, which allow them to retain information across long sequences while mitigating the vanishing gradient problem. Furthermore, **Temporal Convolutional Networks (TCN)** and **CNN-LSTM** hybrids were evaluated for their capacity to capture local patterns and hierarchical temporal structures. These models were benchmarked against the classical machine learning models established in the previous stage.

### 10.2 Evolution of Deep Learning Datasets

The transition from the initial deep learning dataset (`dataset_v2`) to the final configuration (`dataset_v3`) was driven by critical convergence issues identified during early experimentation.

#### 10.2.1 Dataset V2 Characteristics
*   **Source**: `engineered_features.parquet`
*   **Sequence Window**: $L = 96$ (24 hours).
*   **Input Features**: Standardized (mean $\approx 0$, std $\approx 1$).
*   **Target Variable**: Kept in original Megawatt (MW) scale.

#### 10.2.2 The Convergence Problem
During training on `dataset_v2`, recurrent architectures (LSTM and GRU) failed to converge. Diagnosis revealed a significant **scale imbalance** between the input features and the target variable:
*   **Inputs**: $\mu \approx 0, \sigma \approx 1$.
*   **Target**: $\mu \approx 6291$ MW, $\sigma \approx 1891$ MW.

This magnitude gap (four orders of magnitude) caused gradient instability, as the loss function's sensitivity to weights in the final linear layer was disproportionately high, leading to rapidly exploding or vanishing gradients during backpropagation through time.

### 10.3 Construction of `dataset_v3_sequence_dataset`

To resolve the scale imbalance and enhance temporal coverage, a redesigned pipeline was implemented followed a strict processing order:

1.  **Loading and Sorting**: Data was ingested from the engineered feature matrix and sorted chronologically to ensure no temporal shuffling.
2.  **Numerical Sanitization**: Infinite values were replaced with NaNs and repaired using bidirectional filling (`ffill()` followed by `bfill()`), ensuring a continuous feature matrix.
3.  **Chronological Partitioning**: The final **35,040 samples** (exactly one year) were reserved for testing to provide a comprehensive seasonal evaluation.
4.  **Feature Filtering**: The raw `load_MW` (target), `hour`, and `day_of_week` columns were removed. Cyclical sine/cosine encodings were retained to provide continuous temporal context.
5.  **Zero Variance Filtering**: A `VarianceThreshold` filter was applied to remove features with near-zero variance, reducing dimensionality and noise.
6.  **Independent Scaling**: Two distinct `StandardScaler` objects were utilized:
    *   `feature_scaler`: Applied to inputs $\mathbf{X}$.
    *   `y_scaler`: Applied to the target $y$.
    Both were fitted exclusively on the training set to prevent data leakage.
7.  **Window Construction**: The sequence length was expanded to $L = 192$ (48 hours). For each index $i$, the sample is defined as:
    $$\mathbf{X}_i = \{f_i, f_{i+1}, \dots, f_{i+191}\}, \quad y_i = \text{load}_{i+192}$$
    This 48-hour window allows models to capture both short-term momentum and multi-day patterns.
8.  **Tensor Serialization**: All tensors were stored as `float32` numpy arrays.

#### 10.3.1 Final Dataset Dimensions
| Partition | Sequences | Timesteps | Features |
| :--- | :--- | :--- | :--- |
| **Train** | 107,327 | 192 | 45 |
| **Test** | 34,848 | 192 | 45 |

### 10.4 Deep Learning Model Architectures

Four distinct architectures were implemented and evaluated:

*   **LSTM**: Two stacked layers with hidden units of 64 and 32 respectively. Dropout (0.2) was applied between layers to prevent overfitting.
*   **GRU**: Similar to the LSTM, utilizing two stacked layers (64 $\rightarrow$ 32) with a dropout rate of 0.2.
*   **CNN-LSTM**: A hybrid architecture utilizing two 1D-convolutional layers (32 filters, kernel size = 3, ReLU activation) for local feature extraction, followed by a single LSTM layer (64 units).
*   **Temporal Convolutional Network (TCN)**: A pure convolutional approach using four levels of dilated causal convolutions (dilations: 1, 2, 4, 8) with a kernel size of 3 and residual skip connections.

### 10.5 Training Pipeline & Configuration

The models were trained using a standardized pipeline to ensure comparability:

*   **Optimizer**: Adam ($\eta = 0.001$).
*   **Loss Function**: Mean Squared Error (MSE).
*   **Batch Size**: 64.
*   **Regularization**: Gradient clipping (`max_norm = 1.0`) and Early Stopping (patience = 10 epochs).
*   **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=5) monitored on validation loss.

**Evaluation**: Predictions were inverse-transformed using the `y_scaler` prior to metric calculation to ensure all errors (MAE, RMSE, MAPE) are reported in original Megawatt (MW) units.

## 11. Deep Learning Experimental Results

The performance of the sequence-based models on `dataset_v3` is summarized in Table 2.

**Table 2: Deep Learning Benchmark Results (Dataset v3)**

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **GRU** | 144.53 | ~211 | 2.00 |
| **LSTM** | 153.00 | ~219 | 2.15 |
| **CNN-LSTM** | Failed to Converge | — | — |
| **TCN** | Failed to Converge | — | — |

### 11.1 Prior Experiment: TCN Performance on Dataset V2
It is noteworthy that in earlier experiments using `dataset_v2` (unscaled target, $L=96$), the TCN achieved a competitive **MAPE of 2.79%**. The subsequent failure of convolutional architectures (CNN-LSTM and TCN) to converge on `dataset_v3` suggests that the increased sequence length and target scaling may have altered the loss surface in a way that is less conducive to convolutional optimization compared to recurrent gates.

## 12. Interpretation and Comparative Analysis

### 12.1 Recurrent Model Performance
The GRU emerged as the superior deep learning model with a MAPE of **2.00%**. While this represents strong predictive capability, it remains inferior to the results achieved by classical models like LightGBM (**1.79%**) and the Persistence baseline (**1.63%**).

### 12.2 Why Classical Models Outperform Deep Learning
For the Haryana load dataset, the results suggest that feature-engineered tree-based models provide stronger predictive performance than deep sequential models for several reasons:

1.  **Hand-crafted Temporal Context**: The engineered lag features and rolling statistics already encode the temporal dependencies that deep models are tasked with learning from scratch.
2.  **Structural Regularity**: Electricity demand patterns are highly structured and periodic. Tree-based models handle tabularized historical features with high efficiency.
3.  **Data Resolution**: At 15-minute intervals, the series exhibits high inertia. The predictive signal is dominated by immediate history, which classical models capture effectively through simple lag features.

## 13. Reproducibility

To ensure the reproducibility of these results, the following parameters were strictly maintained:
*   **Dataset Version**: `v3_sequence_dataset`
*   **Sequence Length**: 192 (48 hours)
*   **Feature Count**: 45
*   **Tensor Dtype**: `float32`
*   **Random Seed**: 42

All model architectures, training logs, and dataset summaries are archived within the repository.

## 14. Stage 4: Transformer Benchmarking Methodology

### 14.1 Rationale for Using Transformers
Transformer-based architectures were introduced in Stage 4 to evaluate the efficacy of self-attention mechanisms in capturing long-range temporal dependencies within the Haryana electricity load series. While classical recurrent models like LSTM and GRU process sequences iteratively, transformers utilize parallel processing through multi-head attention, enabling the model to assign dynamic weights to different historical timesteps. This capability is theorized to be highly beneficial for Short-Term Load Forecasting (STLF), where specific periodic events or sudden shifts in demand require context from varying temporal distances.

Two specific architectures were selected for this stage:
1.  **Baseline Transformer Encoder**: A standard attention-based encoder to establish a performance floor for transformer-based forecasting.
2.  **PatchTST (Patch Time-Series Transformer)**: A state-of-the-art architecture that segments the time series into overlapping patches, designed to improve temporal abstraction and stability.

The primary objective was to determine whether attention-based modeling could surpass the high performance benchmarks established by LightGBM and Recurrent Neural Networks (RNNs) in earlier stages.

## 15. Dataset Specifications for Transformer Experiments

### 15.1 Dataset Characteristics
The transformer experiments utilized the `dataset_v3_sequence_dataset`, which features:
*   **Sequence Length**: 192 timesteps (48 hours).
*   **Feature Count**: 45 engineered predictors.
*   **Target Variable**: Standardized `load_MW`.

### 15.2 Computational Optimization via Truncation
To accommodate training on a CPU-based research environment, the supervised training set was strategically truncated. 
*   **Original Training Pool**: 107,327 sequences.
*   **Truncated Training Subset**: Capped at 30,000 samples.
*   **Validation Schema**: An 80/20 split was applied to this 30,000-sample subset (24,000 train / 6,000 validation).
*   **Test Integrity**: The test partition (34,848 sequences) remained completely untouched, ensuring that benchmarking results remain directly comparable across all pipeline stages.

Scaling was handled via `StandardScaler` fitted exclusively on the training subset to prevent look-ahead bias.

## 16. Baseline Transformer Implementation

The baseline architecture consists of a linear embedding layer followed by a stack of transformer encoder layers. This model processes the full sequence of 192 timesteps directly.

### 16.1 Model Configuration
| Parameter | Value |
| :--- | :--- |
| Sequence Length | 192 |
| Feature Count | 45 |
| d_model (Embedding Dim) | 64 |
| Attention Heads | 4 |
| Encoder Layers | 4 |
| Dropout Rate | 0.1 |
| Output Head | Linear Regression |

Multi-head attention partitioning was ensured by enforcing that `d_model % num_heads == 0`.

## 17. PatchTST Architecture

PatchTST introduces a "patching" mechanism to the transformer encoder. Instead of processing individual timesteps, the model divides the 48-hour sequence into temporal patches, reducing the effective sequence length and focusing the attention mechanism on higher-level temporal patterns.

### 17.1 PatchTST Configuration
| Parameter | Value |
| :--- | :--- |
| Patch Length | 16 |
| Patch Stride | 8 |
| Channel Independence | True |
| d_model | 64 |
| Attention Heads | 4 |
| Encoder Layers | 4 |
| Sequence Length | 192 |

The channel-independent strategy ensures that each feature channel is processed separately by the shared encoder stack before the results are flattened and aggregated by the regression head.

## 18. Training Configuration and Stability

The training pipeline for Stage 4 was designed for numerical stability and reproducibility.

*   **Optimizer**: Adam ($\eta = 0.0005$).
*   **Batch Size**: 128.
*   **Epochs**: 30 (with early stopping after 5 epochs of no improvement).
*   **Stability Mechanisms**: Gradient clipping was enforced at a maximum norm of 1.0 to prevent gradient explosion.
*   **Reproduction**: Deterministic seeds (42) were set for `torch`, `numpy`, and `random`.

## 19. Evaluation Framework

All predictions generated by the transformer models were **inverse-transformed** using the `y_scaler` metadata before metric calculation. This ensures that the final results are reported in original Megawatt (MW) units, maintaining consistency with Stage 1 (Classical ML) and Stage 2 (RNNs).

Evaluation metrics include:
*   **MAE**: Mean Absolute Error (MW).
*   **RMSE**: Root Mean Squared Error (MW).
*   **MAPE**: Mean Absolute Percentage Error (%).

## 20. Transformer Benchmarking Results

The performance of the attention-based models is summarized in Table 3.

**Table 3: Transformer Benchmarking Results**
| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| Baseline Transformer | 182.41 | 269.08 | 2.45 |
| PatchTST | 251.90 | 352.56 | 3.58 |

### 20.1 Computational Cost Analysis
Transformer architectures exhibited significantly higher computational requirements compared to earlier stages.
*   **Hardware**: Experiments were conducted on a standard CPU environment.
*   **Training Time**: Despite truncation to 30,000 samples and a reduced `d_model`, training sessions for PatchTST reached approximately 5.3 hours (19,174 seconds).
*   **Performance Trade-off**: The introduction of PatchTST resulted in a substantial increase in training time without a corresponding improvement in accuracy on this specific dataset.

## 21. Interpretation and Comparative Analysis

### 21.1 Cross-Stage Comparison
The performance of transformers relative to earlier model families is illustrated in Table 4.

**Table 4: Comparative Results Across All Pipeline Stages**
| Model | Model Type | MAPE (%) |
| :--- | :--- | :---: |
| **LightGBM** | Classical ML (GBDT) | **1.78** |
| **GRU** | Recurrent Neural Network | **2.00** |
| **LSTM** | Recurrent Neural Network | 2.15 |
| **Baseline Transformer** | Attention-based Model | 2.45 |
| **PatchTST** | Patch-based Transformer | 3.58 |

### 21.2 Analysis of Transformer Performance
The benchmarking results indicate that while transformer architectures are capable of capturing temporal dynamics, they do not outperform a well-tuned LightGBM model on the Haryana dataset. 
1.  **Feature Interaction Sensitivity**: PatchTST’s underperformance (3.58% MAPE) is likely due to the **channel-independent** assumption. In electricity forecasting, interactions between temporal features (e.g., hour of day) and structural features (e.g., rolling means) are critical; processing these independently restricts the model's ability to learn these interactions.
2.  **Redundancy with Engineered Features**: Much of the "context" transformers are designed to learn is already explicitly provided to the models through the 45 engineered features in `dataset_v3`. This suggests that for structured time series with heavy feature engineering, attention mechanisms may offer diminishing returns.

## 22. Key Findings

1.  **Baseline Competitiveness**: The Baseline Transformer achieved a MAPE of 2.45%, demonstrating robust but non-superior forecasting capability.
2.  **Structural Assumption Mismatch**: The channel-independent design of PatchTST is poorly suited for datasets where complex inter-feature interactions are the primary predictive signal.
3.  **Classical ML Dominance**: Gradient Boosted Trees (LightGBM) remains the optimal choice for the Haryana STLF problem due to its superior accuracy and significantly lower computational cost.

## 23. Stage 4 Reproducibility

### 23.1 Strict Parameter Sourcing
To ensure scientific integrity, all parameters documented for Stage 4 were sourced strictly from controlled implementation artifacts:
*   Configurations were derived from `experiments/transformers/config.py`.
*   Sequence metadata was extracted from `dataset_summary.json`.
*   Final metrics and training times were cross-referenced with `transformer_model_results.csv`.

### 23.2 Required Artifacts
Replication of Stage 4 requires the `dataset_v3_sequence_dataset` (30k truncated subset) and the specific deterministic seeds defined in the training pipeline.

## 24. Stage 5: Hybrid Residual Forecasting Pipeline

### 24.1 Motivation for Hybrid Forecasting
Stage 5 introduces a hybrid forecasting framework designed to leverage the complementary strengths of gradient boosting and recurrent neural networks. Classical machine learning models, specifically LightGBM, have demonstrated superior performance in capturing nonlinear interactions within the high-dimensional hand-crafted feature space. However, such models treat timesteps as independent tabular rows, potentially overlooking subtle sequential dependencies within the error patterns.

The hybrid architecture aims to model the systematic errors of the tree-based model using a Gated Recurrent Unit (GRU) network. By learning the structure of the residuals, the GRU serves as an additive corrective signal, theoretically refining the baseline forecast.

The mathematical formulation for the hybrid framework is defined as follows:
Let $y_t$ represent the true electricity load at time $t$, and $\hat{y}_{LGBM}(t)$ represent the prediction from the optimized LightGBM baseline. 

The **residual** ($r_t$) is defined as the difference between the ground truth and the baseline prediction:
$$r_t = y_t - \hat{y}_{LGBM}(t)$$

The GRU model is then trained to predict this residual based on the historical sequence $x_t$:
$$\hat{r}_t = \text{GRU}(x_t)$$

The final architecture computes the **hybrid forecast** ($\hat{y}_{final}$) as:
$$\hat{y}_{final}(t) = \hat{y}_{LGBM}(t) + \hat{r}_t$$

## 25. Dataset Used in Hybrid Experiment

### 25.1 Data Source and Properties
The hybrid experiments reused the `dataset_v3_sequence_dataset` to ensure benchmark consistency across all deep learning and attention-based stages. 
*   **Sequence Length**: 192 timesteps (48 hours).
*   **Feature Count**: 45 engineered predictors.
*   **Target Variable**: Residuals derived from the LightGBM baseline.

### 25.2 Dataset Partitioning
The chronological partition of `dataset_v3` was strictly maintained:
*   **Training Sequences**: 107,327.
*   **Test Sequences**: 34,848.

No modifications were made to the sequence construction, ensuring that the input feature space $x_t$ remains identical to the Stage 2 (RNN) and Stage 4 (Transformer) pipelines.

## 26. LightGBM Baseline Model

The classical ML baseline utilized for residual generation is the LightGBM model trained in Stage 1. 
*   **Model Source**: `lightgbm.joblib`.
*   **Feature Schema**: Feature ordering was strictly enforced using `feature_list.json` to ensure consistency with the original training environment.
*   **Inference**: The model instance was reused without retraining to prevent any inconsistency between the residual learning target and the reported Stage 1 performance.

## 27. Residual Target Construction

### 27.1 Residual Generation Pipeline
A multi-step pipeline was implemented to generate and align the residual training targets:
1.  **Feature Ingestion**: Loading the original `engineered_features.parquet`.
2.  **Schema Enforcement**: Reordering columns to match the 45-feature schema defined in `feature_list.json`.
3.  **Baseline Prediction**: Generating $\hat{y}_{LGBM}(t)$ for all historical points.
4.  **Residual Computation**: Calculating $r_t = y_t - \hat{y}_{LGBM}(t)$.

### 27.2 Timestamp-Based Alignment
To ensure perfect synchronization between the existing `dataset_v3` sequence tensors and the new residual targets, a **timestamp-based lookup** was utilized:
*   The original sequence target timestamps were loaded from `timestamps_train.npy` and `timestamps_test.npy`.
*   Residual values were mapped to each sequence index using these timestamps, ensuring that $\hat{r}_i$ corresponds exactly to the temporal window represented by $\mathbf{X}_i$.
*   This method guarantees that the residual model learns the specific error incurred by LightGBM at the exact timestep it is tasked with forecasting.

### 27.3 Residual Diagnostic Statistics
Prior to training, the residual signal was validated for statistical stability. The metrics harvested from the residual generation script are summarized in Table 5.

**Table 5: Residual Signal Validation Statistics**
| Metric | Training Set | Test Set |
| :--- | :---: | :---: |
| **Mean** | -0.0066 | 5.8504 |
| **Std. Deviation** | 170.20 | 195.40 |
| **Max Absolute Residual** | 5139.57 | 3330.51 |
| **95th Percentile Abs.** | 346.67 | 403.51 |

The low mean residual relative to the standard deviation confirms that the LightGBM baseline is unbiased, while the residual variance provides sufficient signal for sequential learning.

## 28. GRU Residual Model Architecture

The corrective model, implemented in `gru_residual_model.py`, utilizes a many-to-one recurrent architecture:
*   **Input Layer**: Accepts sequence tensors of shape (192, 45).
*   **Recurrent Stack**: Two stacked GRU layers with hidden dimensions of 64 and 32 respectively.
*   **Internal Normalization**: A `LayerNorm` layer is applied after the first GRU pass for gradient stability.
*   **Regularization**: Dropout ($p=0.2$) is applied to the output of the final recurrent state.
*   **Output Head**: A single fully connected linear layer mapping the 32-dimensional hidden state to a scalar prediction $\hat{r}_t$.

## 29. Training Configuration

The GRU network was trained to minimize the Mean Squared Error (MSE) on the residual targets.
*   **Batch Size**: 128.
*   **Learning Rate**: 0.001 (initial).
*   **Epochs**: 30 (with periodic early stopping).
*   **Optimization Strategy**: Adam optimizer combined with a `ReduceLROnPlateau` scheduler (factor=0.5, patience=5).
*   **Numerical Stability**: Gradient norm clipping was enforced at 1.0.
*   **Validation**: A chronological 80/20 split of the 107,327 training sequences was used to monitor convergence.

## 30. Hybrid Inference Pipeline

Evaluation was performed using a three-stage wall-clock timed inference loop:
1.  **Baseline Path**: Compute $\hat{y}_{LGBM}$ using the LightGBM model.
2.  **Corrective Path**: Compute $\hat{r}$ using the GRU model based on the 48-hour historical window.
3.  **Synthesis**: Final forecast $\hat{y}_{final} = \hat{y}_{LGBM} + \hat{r}$.

Both signals are computed in Megawatt scale; consequently, no inverse scaling is required at the final synthesis stage.

## 31. Evaluation Metrics

To ensure cross-stage comparability, the hybrid model was evaluated using the same standardized metrics:
*   **MAE**: Mean Absolute Error.
*   **RMSE**: Root Mean Squared Error.
*   **MAPE**: Mean Absolute Percentage Error.

## 32. Hybrid Model Results

The performance of the LightGBM+GRU hybrid pipeline is summarized in Table 6.

**Table 6: Hybrid Residual Model Performance (Dataset v3)**
| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
| :--- | :--- | :--- | :---: |
| **LightGBM_GRU_Hybrid** | 129.43 | 195.54 | 1.79 |

### 32.1 Computational Performance
The hybrid inference and training costs were recorded using wall-clock timing:
*   **Training Time**: 116.9 seconds (30 epochs on CPU).
*   **Inference Time**: 14.13 seconds (test set evaluation).

The relatively high inference time is primarily driven by the recurrent processing of the 34,848 test sequences, which incurs significantly more overhead than the baseline LightGBM model.

## 33. Interpretation of Hybrid Results

The hybrid model achieved a MAPE of 1.79%, which is nearly identical to the original LightGBM baseline (1.78%). This result yields several critical insights:
1.  **Temporal Signal Saturation**: The fact that a deep recurrent model could not significantly reduce LightGBM's residual error suggests that the 45 hand-crafted features already capture the vast majority of the temporal dynamics present in the Haryana grid.
2.  **Low Modelable Variance**: Residual analysis indicates that LightGBM's errors are primarily driven by stochastic variations rather than systematic sequential patterns that a GRU could predictably model.
3.  **Additive Signal Limits**: The additive residual approach provides marginal benefits when the feature-engineered baseline is already highly optimized.

## 34. Research Implications

The Hybrid Pipeline experiment reinforces the findings from Stage 1: for structured, high-resolution electricity demand data, extensive feature engineering projects the problem into a space where tree-based ensembles are nearly optimal. While deep learning models (Stage 2) and attention-based models (Stage 4) exhibit strong predictive power, their ability to refine a well-structured tree model through residual learning is limited in this specific dataset.

## 35. Reproducibility Details

### 35.1 Strict Parameter Sourcing
*   **Configurations**: Defined in `experiments/hybrid_models/config.py`.
*   **Architecture**: Specified in `gru_residual_model.py`.
*   **Logic**: Implemented in `run_hybrid_benchmark.py` and `residual_dataset_builder.py`.

### 35.2 Directory Structure and Artifacts
All artifacts required for replication are archived under:
`experiments/hybrid_models/`
Required files include `lightgbm.joblib`, `feature_list.json`, and the `.npy` timestamp files for sequence alignment.
