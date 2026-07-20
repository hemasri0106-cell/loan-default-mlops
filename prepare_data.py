import os
import sys
import time
import glob
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
import psutil
import kagglehub

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# List of columns to load
COLUMNS_TO_LOAD = [
    'loan_amnt',
    'term',
    'int_rate',
    'installment',
    'grade',
    'sub_grade',
    'emp_length',
    'home_ownership',
    'annual_inc',
    'verification_status',
    'purpose',
    'addr_state',
    'dti',
    'fico_range_low',
    'fico_range_high',
    'open_acc',
    'revol_util',
    'issue_d',
    'loan_status'
]

# Categorical columns
CATEGORICAL_COLUMNS = [
    'grade',
    'sub_grade',
    'term',
    'purpose',
    'home_ownership',
    'verification_status',
    'addr_state',
    'emp_length'
]

def get_current_memory_mb():
    """Gets current RSS memory of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.update()

    def update(self):
        mem = self.process.memory_info().rss
        if mem > self.peak_memory:
            self.peak_memory = mem
        return mem

    def get_peak_mb(self):
        return self.peak_memory / (1024 * 1024)

def download_dataset():
    """Downloads the lending-club dataset using kagglehub and locates the accepted loans file."""
    logger.info("Downloading dataset using kagglehub...")
    download_start = time.time()
    
    # Download using kagglehub
    dataset_path = kagglehub.dataset_download("wordsforthewise/lending-club")
    logger.info(f"Dataset downloaded to: {dataset_path} in {time.time() - download_start:.2f} seconds")
    
    # Search for files matching accepted*.csv* using recursive glob
    search_pattern = os.path.join(dataset_path, "**", "accepted*.csv*")
    matching_files = glob.glob(search_pattern, recursive=True)
    
    # Filter to only actual files
    files = [f for f in matching_files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"Could not find any files matching {search_pattern}")
    
    # Prefer uncompressed .csv over compressed if available
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        file_path = csv_files[0]
    else:
        file_path = files[0]
        
    logger.info(f"Locating accepted loans dataset: {file_path}")
    return file_path

def process_chunk(chunk):
    """Processes a single chunk of data."""
    # 1. Filter rows by loan_status
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    chunk = chunk[chunk['loan_status'].isin(valid_statuses)].copy()
    if chunk.empty:
        return chunk
    
    # 2. Convert loan_status to binary target (int32): Fully Paid -> 0, Charged Off/Default -> 1
    target_map = {
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    }
    chunk['loan_status'] = chunk['loan_status'].astype(str).map(target_map).astype('int32')
    
    # 3. Clean percentage columns (int_rate and revol_util)
    for col in ['int_rate', 'revol_util']:
        if col in chunk.columns:
            # Strip '%' and convert to float32
            chunk[col] = chunk[col].astype(str).str.rstrip('%')
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')
            
    # 4. Parse issue_d and extract issue_year
    chunk['issue_d'] = pd.to_datetime(chunk['issue_d'], format="%b-%Y", errors="coerce")
    chunk['issue_year'] = chunk['issue_d'].dt.year
    
    # Drop rows where issue_year cannot be parsed
    chunk = chunk.dropna(subset=['issue_year']).copy()
    chunk['issue_year'] = chunk['issue_year'].astype('int32')
    
    # 5. Type cast numeric columns to float32
    numeric_cols = [
        'loan_amnt', 'installment', 'annual_inc', 'dti', 
        'fico_range_low', 'fico_range_high', 'open_acc'
    ]
    for col in numeric_cols:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')
            
    return chunk

def stratified_sample(df, target_n=20000, random_state=42):
    """
    Performs manual stratified sampling on 'loan_status' to preserve the exact class distribution
    and sample exactly target_n rows.
    """
    if len(df) <= target_n:
        return df
    
    # Class distribution counts
    class_counts = df['loan_status'].value_counts()
    total_rows = len(df)
    
    # Calculate sample sizes using truncation to prevent rounding overshoot
    target_sizes = {label: int(count * target_n / total_rows) for label, count in class_counts.items()}
    
    # Calculate discrepancy due to truncation
    diff = target_n - sum(target_sizes.values())
    
    # Distribute the difference to classes with the largest fractional remainders
    if diff > 0:
        fractional_parts = {
            label: (count * target_n / total_rows) - target_sizes[label] 
            for label, count in class_counts.items()
        }
        # Sort classes by their fractional part descending
        sorted_labels = sorted(fractional_parts.keys(), key=lambda l: fractional_parts[l], reverse=True)
        for i in range(diff):
            target_sizes[sorted_labels[i]] += 1
            
    # Sample from each class
    sampled_dfs = []
    for label, size in target_sizes.items():
        class_subset = df[df['loan_status'] == label]
        if size > 0:
            sampled_dfs.append(class_subset.sample(n=size, random_state=random_state))
            
    return pd.concat(sampled_dfs)

def main():
    start_time = time.time()
    mem_tracker = MemoryTracker()
    
    try:
        # Step 1: Download dataset
        file_path = download_dataset()
        mem_tracker.update()
        
        # Step 2: Read CSV in chunks
        chunksize = 25000
        logger.info(f"Processing dataset in chunks of size {chunksize}...")
        
        original_rows = 0
        filtered_rows = 0
        yearly_dfs = defaultdict(list)
        
        # Setup data types to load efficiently in read_csv where possible
        # Note: Columns to clean (int_rate, revol_util) or parse (issue_d) are loaded as object/string first
        dtype_mapping = {
            'loan_amnt': 'float32',
            'term': 'category',
            'installment': 'float32',
            'grade': 'category',
            'sub_grade': 'category',
            'emp_length': 'category',
            'home_ownership': 'category',
            'annual_inc': 'float32',
            'verification_status': 'category',
            'purpose': 'category',
            'addr_state': 'category',
            'dti': 'float32',
            'fico_range_low': 'float32',
            'fico_range_high': 'float32',
            'open_acc': 'float32',
            'loan_status': 'category',
            'int_rate': 'object',
            'revol_util': 'object',
            'issue_d': 'object'
        }
        
        # Read the csv chunk by chunk
        # Set engine='c' for performance, low_memory=False to suppress warnings
        chunks = pd.read_csv(
            file_path,
            usecols=COLUMNS_TO_LOAD,
            dtype=dtype_mapping,
            chunksize=chunksize,
            low_memory=False
        )
        
        for idx, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            original_rows += chunk_len
            
            # Process the chunk
            processed = process_chunk(chunk)
            filtered_rows += len(processed)
            
            # Group rows by issue_year and store in dictionary
            if not processed.empty:
                for year, group in processed.groupby('issue_year'):
                    yearly_dfs[year].append(group)
                    
            if (idx + 1) % 10 == 0:
                mem_tracker.update()
                logger.info(f"Processed {(idx + 1) * chunksize:,} rows... Peak RAM: {mem_tracker.get_peak_mb():.2f} MB")
                
        mem_tracker.update()
        logger.info("Finished reading all chunks.")
        logger.info(f"Original rows processed: {original_rows:,}")
        logger.info(f"Rows after filtering: {filtered_rows:,}")
        
        # Step 3: Stratified Sampling per year
        logger.info("Performing yearly stratified sampling...")
        yearly_available_counts = {}
        yearly_sampled_counts = {}
        sampled_year_dfs = []
        
        for year, dfs in yearly_dfs.items():
            # Concatenate all chunks for this year
            year_df = pd.concat(dfs, ignore_index=True)
            # Ensure categorical columns are properly categorical
            for col in CATEGORICAL_COLUMNS:
                if col in year_df.columns:
                    year_df[col] = year_df[col].astype('category')
                    
            num_rows = len(year_df)
            yearly_available_counts[year] = num_rows
            
            # Perform stratified sampling
            sampled_df = stratified_sample(year_df, target_n=20000, random_state=42)
            yearly_sampled_counts[year] = len(sampled_df)
            sampled_year_dfs.append(sampled_df)
            
            mem_tracker.update()
            
        # Step 4: Combine, Shuffle and Reset Index
        logger.info("Combining and shuffling final dataset...")
        final_df = pd.concat(sampled_year_dfs, ignore_index=True)
        
        # Explicitly cast all categorical columns to category to ensure final types are preserved
        for col in CATEGORICAL_COLUMNS:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype('category')
                
        final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Step 5: Save processed dataset
        output_dir = os.path.join("data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "loan_default_processed.parquet")
        
        logger.info(f"Saving final dataset to {output_path}...")
        final_df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info("Dataset saved successfully.")
        
        # Step 6: Print reports
        total_time = time.time() - start_time
        peak_mem = mem_tracker.get_peak_mb()
        
        print("\n" + "="*60)
        print("           LOAN DEFAULT DATA PREPROCESSING REPORT")
        print("="*60)
        print(f"1. Original rows processed: {original_rows:,}")
        print(f"2. Rows after filtering: {filtered_rows:,}")
        print("\n3. Number of rows available for every year:")
        for year in sorted(yearly_available_counts.keys()):
            print(f"   - Year {year}: {yearly_available_counts[year]:,}")
        print("\n4. Number of sampled rows for every year:")
        for year in sorted(yearly_sampled_counts.keys()):
            print(f"   - Year {year}: {yearly_sampled_counts[year]:,}")
        print("\n5. Missing values per column (Final Dataframe):")
        missing_vals = final_df.isnull().sum()
        for col, val in missing_vals.items():
            print(f"   - {col:<25}: {val:,}")
        print("\n6. Binary class distribution (Final Dataframe):")
        class_dist = final_df['loan_status'].value_counts()
        for cls, count in class_dist.items():
            prop = count / len(final_df) * 100
            label = "Default/Charged Off (1)" if cls == 1 else "Fully Paid (0)"
            print(f"   - {label:<25}: {count:,} ({prop:.2f}%)")
        print(f"\n7. Final dataframe shape: {final_df.shape}")
        print(f"8. Total processing time: {total_time:.2f} seconds")
        print(f"9. Peak memory usage: {peak_mem:.2f} MB")
        
        print("\n" + "="*60)
        print("          FINAL DATAFRAME MEMORY USAGE & DATA TYPES")
        print("="*60)
        dtypes = final_df.dtypes
        mem_usage = final_df.memory_usage(deep=True) / (1024 * 1024)  # in MB
        print(f"{'Column':<25} | {'Dtype':<15} | {'Memory (MB)':<12}")
        print("-" * 58)
        for col in final_df.columns:
            print(f"{col:<25} | {str(dtypes[col]):<15} | {mem_usage[col]:.4f}")
        print("-" * 58)
        total_df_mem = final_df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"{'Total Memory Usage (MB)':<25} | {'':<15} | {total_df_mem:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.exception("An error occurred during dataset preprocessing:")
        sys.exit(1)

if __name__ == "__main__":
    main()
