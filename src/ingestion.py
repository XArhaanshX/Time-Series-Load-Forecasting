import pandas as pd
import glob
import os
import logging
from tqdm import tqdm
from datetime import datetime, time, timedelta

logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ingest_daily_log(file_path, config):
    """
    Ingests Haryana State Load from a single Excel log.
    Uses dynamic row discovery to handle minor structure variations.
    """
    try:
        # Load first 20 rows to find metadata and headers
        df_full = pd.read_excel(file_path, header=None, nrows=20, engine='openpyxl')
        
        # 1. Discover Date
        report_date = None
        for i in range(len(df_full)):
            row_values = [str(val).strip().upper() for val in df_full.iloc[i].values]
            if "REPORT FOR THE DAY" in row_values:
                # Expecting date in the same row, often index 7 or 8
                for val in df_full.iloc[i].values:
                    if isinstance(val, (datetime, pd.Timestamp)):
                        report_date = val.date()
                        break
                if report_date: break
        
        if not report_date:
            logger.warning(f"Could not find 'REPORT FOR THE DAY' date in {file_path}")
            return None

        # 2. Discover Header Row
        header_row_idx = None
        for i in range(len(df_full)):
            row_values = [str(val).strip().upper() for val in df_full.iloc[i].values]
            if "TIME" in row_values and any("LOAD" in val for val in row_values):
                header_row_idx = i
                break
        
        if header_row_idx is None:
            logger.warning(f"Could not find header row in {file_path}")
            return None

        # 3. Read data
        # We read 96 rows starting after the discovered header
        df = pd.read_excel(
            file_path, 
            header=header_row_idx, 
            nrows=config['data']['excel']['num_rows'],
            usecols=[0, 4], # TIME and Haryana Load (MW) column mapping
            engine='openpyxl'
        )
        
        df.columns = ['TIME', 'load_MW']
        
        def parse_timestamp(t):
            try:
                if isinstance(t, str):
                    t_str = t.strip()
                    if t_str.upper() == "TIME": return pd.NaT
                    try:
                        parsed_t = datetime.strptime(t_str, '%H:%M:%S').time()
                    except:
                        parsed_t = pd.to_datetime(t_str).time()
                elif isinstance(t, datetime):
                    parsed_t = t.time()
                elif isinstance(t, time):
                    parsed_t = t
                else:
                    return pd.NaT
                
                # Combine with report_date
                # IMPORTANT: If time is 00:00:00, it might be the start of the next day 
                # depending on the log convention. But for now we stick to report_date.
                return pd.Timestamp(datetime.combine(report_date, parsed_t))
            except:
                return pd.NaT

        df['timestamp'] = df['TIME'].apply(parse_timestamp)
        df = df.dropna(subset=['timestamp', 'load_MW'])
        
        # Ensure numeric load
        df['load_MW'] = pd.to_numeric(df['load_MW'], errors='coerce')
        df = df.dropna(subset=['load_MW'])
        
        # Filter out obvious junk rows (e.g. if district header was read)
        df = df[df['load_MW'] > 0]
        
        return df[['timestamp', 'load_MW']]

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def run_ingestion(config):
    raw_dir = config['paths']['raw_data']
    files = sorted(glob.glob(os.path.join(raw_dir, "*.xlsx")))
    
    logger.info(f"Starting ingestion of {len(files)} files from {raw_dir}")
    
    all_dfs = []
    for f in tqdm(files, desc="Ingesting Excel logs"):
        df = ingest_daily_log(f, config)
        if df is not None and not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        raise ValueError("No data ingested. Check raw data path and file structure.")
        
    logger.info("Concatenating daily dataframes...")
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Final safety filter for timestamp bounds (Pandas range)
    min_date = pd.Timestamp('1900-01-01')
    max_date = pd.Timestamp('2100-01-01')
    master_df = master_df[(master_df['timestamp'] > min_date) & (master_df['timestamp'] < max_date)]
    
    # Sort and remove duplicates
    master_df = master_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    logger.info(f"Ingestion complete. Total rows: {len(master_df)}")
    return master_df
