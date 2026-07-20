import os
import sys
import time
import json
import logging
from typing import Dict, Tuple
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the preprocessed Parquet dataset."""
    logger.info(f"Loading preprocessed dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found at: {filepath}")
    return pd.read_parquet(filepath)

def validate_split(df: pd.DataFrame, split_name: str, expected_years: set) -> None:
    """Validates that a split contains only the expected years. Raises ValueError if validation fails."""
    if df.empty:
        logger.warning(f"{split_name} split is empty. Skipping validation.")
        return
        
    actual_years = set(df['issue_year'].unique())
    unexpected_years = actual_years - expected_years
    if unexpected_years:
        raise ValueError(
            f"Chronological validation failed for {split_name} split! "
            f"Expected years in {expected_years}, but found unexpected years: {unexpected_years}"
        )
    logger.info(f"Chronological validation passed for {split_name} split. Years found: {actual_years}")

def split_dataset(df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Splits the dataset chronologically based on issue_year and performs year checks."""
    logger.info("Splitting dataset chronologically...")
    
    # Define expected years for each split
    train_years = set(range(2007, 2016))  # 2007 to 2015 inclusive
    val_years = {2016}
    test_years = {2017}
    prod_years = {2018}
    
    # Filter subsets
    train_df = df[df['issue_year'].isin(train_years)].copy()
    val_df = df[df['issue_year'].isin(val_years)].copy()
    test_df = df[df['issue_year'].isin(test_years)].copy()
    prod_df = df[df['issue_year'].isin(prod_years)].copy()
    
    # Perform strict chronological validations
    validate_split(train_df, "Train", train_years)
    validate_split(val_df, "Validation", val_years)
    validate_split(test_df, "Test", test_years)
    validate_split(prod_df, "Production", prod_years)
    
    # Helper to separate features (X) and target (y)
    # y is loan_status
    # X contains all columns except loan_status and issue_year
    def extract_x_y(subset_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        y = subset_df[['loan_status']].copy()
        X = subset_df.drop(columns=['loan_status', 'issue_year']).copy()
        return X, y
        
    X_train, y_train = extract_x_y(train_df)
    X_val, y_val = extract_x_y(val_df)
    X_test, y_test = extract_x_y(test_df)
    X_prod, y_prod = extract_x_y(prod_df)
    
    return {
        'train': (X_train, y_train, train_df['issue_year']),
        'val': (X_val, y_val, val_df['issue_year']),
        'test': (X_test, y_test, test_df['issue_year']),
        'prod': (X_prod, y_prod, prod_df['issue_year'])
    }

def save_splits(splits_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]], output_dir: str) -> None:
    """Saves every split dataset separately as a CSV inside output_dir."""
    logger.info(f"Saving splits to directory: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, (X, y, _) in splits_dict.items():
        X_path = os.path.join(output_dir, f"X_{split_name}.csv")
        y_path = os.path.join(output_dir, f"y_{split_name}.csv")
        
        logger.info(f"Saving X_{split_name}.csv ({len(X):,} rows, {X.shape[1]} features)...")
        X.to_csv(X_path, index=False)
        
        logger.info(f"Saving y_{split_name}.csv ({len(y):,} rows)...")
        y.to_csv(y_path, index=False)
        
    logger.info("All splits saved successfully.")

def generate_reports(splits_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]], total_time: float) -> None:
    """Generates a split metadata report, prints it, and saves as JSON in reports/."""
    report = {}
    
    print("\n" + "="*60)
    print("                DATA SPLIT PREPROCESSING REPORT")
    print("="*60)
    print(f"Total execution time: {total_time:.4f} seconds\n")
    
    for name, (X, y, years) in splits_dict.items():
        num_samples = len(X)
        num_features = X.shape[1]
        
        # Calculate default rate (target == 1)
        default_rate = float(y['loan_status'].mean()) if num_samples > 0 else 0.0
        
        # Get year range
        min_year = int(years.min()) if num_samples > 0 else None
        max_year = int(years.max()) if num_samples > 0 else None
        year_range_str = f"{min_year} - {max_year}" if min_year != max_year else f"{min_year}"
        
        # Get missing values per column
        missing_vals = X.isnull().sum().to_dict()
        
        report[name] = {
            "num_samples": num_samples,
            "num_features": num_features,
            "default_rate": default_rate,
            "year_range": [min_year, max_year] if min_year is not None else [],
            "missing_values": missing_vals
        }
        
        print(f"Split: {name.upper()}")
        print(f"  - Year Range   : {year_range_str}")
        print(f"  - Samples      : {num_samples:,}")
        print(f"  - Features     : {num_features}")
        print(f"  - Default Rate : {default_rate:.4f} ({default_rate * 100:.2f}%)")
        print("  - Missing Values per Column:")
        cols_with_missing = {col: count for col, count in missing_vals.items() if count > 0}
        if cols_with_missing:
            for col, count in cols_with_missing.items():
                print(f"    * {col:<25}: {count:,}")
        else:
            print("    * No missing values")
        print("-" * 60)
        
    # Save report to reports/split_report.json
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "split_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved metadata report to {report_path}")
    print("="*60)

def main():
    start_time = time.time()
    
    # Define paths
    input_path = os.path.join("data", "processed", "loan_default_processed.parquet")
    output_dir = os.path.join("data", "splits")
    
    try:
        # Load dataset
        df = load_data(input_path)
        
        # Split dataset & validate
        splits = split_dataset(df)
        
        # Save split files
        save_splits(splits, output_dir)
        
        # Report execution statistics
        total_time = time.time() - start_time
        generate_reports(splits, total_time)
        
    except Exception as e:
        logger.exception("An error occurred during chronological data splitting:")
        sys.exit(1)

if __name__ == "__main__":
    main()
