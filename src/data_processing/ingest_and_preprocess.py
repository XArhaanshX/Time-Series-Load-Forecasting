import pandas as pd
import numpy as np
import glob
import os
import json
import joblib
from datetime import datetime, timedelta, time
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def load_one_file(file_path):
    """
    Strict ingestion logic for a single Excel file.
    """
    try:
        # 1. Read first 10 rows to find the date metadata
        df_meta = pd.read_excel(file_path, header=None, nrows=10)
        report_date = None
        for i, row in df_meta.iterrows():
            row_str = [str(x).strip().lower() for x in row.values]
            if 'report for the day' in row_str:
                for val in row.values:
                    if isinstance(val, (datetime, pd.Timestamp)):
                        report_date = val.date()
                        break
                if report_date: break
        
        if not report_date:
            raise ValueError(f"Could not find 'REPORT FOR THE DAY' date in {file_path}")

        # 2. Strict read: header=5, column 0 (TIME) and 4 (Load)
        df = pd.read_excel(file_path, header=5, usecols=[0, 4])
        df.columns = ['TIME', 'load_MW']
        
        # 3. Truncate to isolate main table (Stop before district tables)
        # District tables start with a new "TIME" header in column 0
        # We look for the first occurrence of "TIME" as a string after the actual header
        time_header_indices = df.index[df['TIME'].astype(str).str.strip().str.upper() == 'TIME'].tolist()
        if time_header_indices:
            # Drop everything from the first "TIME" header onwards
            df = df.iloc[:time_header_indices[0]]
            
        # 4. Time-based Filtering
        # Convert TIME column to time objects, coercing errors to NaT
        def parse_time(t):
            if isinstance(t, time): return t
            if isinstance(t, datetime): return t.time()
            if isinstance(t, str):
                try: 
                    # Try common formats
                    return datetime.strptime(t.strip(), '%H:%M:%S').time()
                except:
                    try: return datetime.strptime(t.strip(), '%H:%M').time()
                    except: return pd.NaT
            return pd.NaT

        df['TIME_clean'] = df['TIME'].apply(parse_time)
        
        # Drop rows where TIME_clean is NaT (excludes headers of other tables and district labels)
        df = df.dropna(subset=['TIME_clean'])
        
        # 4. Data Conversion and Validations
        df['load_MW'] = pd.to_numeric(df['load_MW'], errors='coerce')
        df = df.dropna(subset=['load_MW'])
        
        # File-level Validations
        row_count = len(df)
        if not (90 <= row_count <= 100):
            raise ValueError(f"Invalid row count ({row_count}) in {file_path}. Expected ~96.")

        if (df['load_MW'] < 0).any():
            raise ValueError(f"Negative load detected in {file_path}.")
        
        daily_max = df['load_MW'].max()
        if daily_max < 1000:
            raise ValueError(f"Unrealistic maximum load ({daily_max} MW) in {file_path}. Expected > 1000.")
            
        if (df['load_MW'] < 1500).any() or (df['load_MW'] > 13000).any():
             logger.warning(f"File {file_path} contains values outside 1500-13000 MW range. Min: {df['load_MW'].min()}, Max: {daily_max}")

        # Construct full timestamps
        df['timestamp'] = df['TIME_clean'].apply(lambda t: datetime.combine(report_date, t))
        
        return df[['timestamp', 'load_MW']]

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def load_raw_files(data_dir='data/raw/'):
    files = glob.glob(os.path.join(data_dir, '*.xlsx'))
    logger.info(f"Initiating strict ingestion for {len(files)} files.")
    
    all_data = []
    skipped = 0
    
    for f in tqdm(files, desc="Ingesting files"):
        df = load_one_file(f)
        if df is not None:
            all_data.append(df)
        else:
            skipped += 1
            
    if not all_data:
        raise ValueError("No data could be successfully ingested.")
        
    master_df = pd.concat(all_data, ignore_index=True)
    return master_df, skipped

def preprocess_dataset(df):
    """
    Standard S2 processing with strict order.
    """
    logger.info("Starting Preprocessing (S2)")
    
    # 1. Cleaning and Sorting
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    # 2. Frequent Enrollment
    df = df.set_index('timestamp')
    missing_count = len(pd.date_range(df.index.min(), df.index.max(), freq='15min')) - len(df)
    df = df.asfreq('15min')
    
    # 3. Imputation (3-Step)
    initial_nans = df['load_MW'].isna().sum()
    # Step 1: Forward Fill (limit 4)
    df['load_MW'] = df['load_MW'].ffill(limit=4)
    # Step 2: Seasonal (t-96)
    mask = df['load_MW'].isna()
    if mask.any():
        df.loc[mask, 'load_MW'] = df['load_MW'].shift(96)[mask]
    # Step 3: Rolling mean fallback
    mask = df['load_MW'].isna()
    if mask.any():
        df.loc[mask, 'load_MW'] = df['load_MW'].rolling(window=8, min_periods=1, center=True).mean()[mask]
    df['load_MW'] = df['load_MW'].bfill() # Final safety
    
    # 4. Outliers
    q1, q3 = df['load_MW'].quantile(0.25), df['load_MW'].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers_mask = (df['load_MW'] < lower) | (df['load_MW'] > upper)
    outliers_count = outliers_mask.sum()
    if outliers_count > 0:
        df.loc[outliers_mask, 'load_MW'] = df['load_MW'].rolling(window=8, min_periods=1, center=True).median()[outliers_mask]
        
    # 5. Normalization
    scaler = MinMaxScaler()
    df['load_scaled'] = scaler.fit_transform(df[['load_MW']])
    
    # Save Scaler
    os.makedirs('models/scalers', exist_ok=True)
    joblib.dump(scaler, 'models/scalers/load_scaler.pkl')
    
    return df, missing_count, outliers_count, initial_nans

def main():
    try:
        raw_df, skipped_files = load_raw_files()
        
        processed_df, missing, outliers, imputed = preprocess_dataset(raw_df)
        
        # Save Outputs
        os.makedirs('data/processed', exist_ok=True)
        processed_df.reset_index().to_csv('data/processed/clean_load_dataset.csv', columns=['timestamp', 'load_MW'], index=False)
        processed_df.reset_index().to_csv('data/processed/processed_dataset.csv', columns=['timestamp', 'load_MW', 'load_scaled'], index=False)
        
        # Reporting
        report = {
            "files_read_success": int(len(raw_df.groupby(raw_df['timestamp'].dt.date))), # Approximation
            "files_skipped": int(skipped_files),
            "total_rows_final": int(len(processed_df)),
            "missing_timestamps_inserted": int(missing),
            "outliers_replaced": int(outliers),
            "status": "Success"
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info("Pipeline Execution Successful.")
        print(json.dumps(report, indent=4))
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
