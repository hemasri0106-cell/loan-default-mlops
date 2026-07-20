import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_datasets(splits_dir: str) -> Dict[str, pd.DataFrame]:
    """Loads feature datasets (X_train, X_val, X_test, X_prod) from CSV files."""
    datasets = {}
    splits = ["train", "val", "test", "prod"]
    
    for split in splits:
        filename = f"X_{split}.csv"
        filepath = os.path.join(splits_dir, filename)
        logger.info(f"Loading dataset: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required split file not found: {filepath}")
        datasets[split] = pd.read_csv(filepath)
        
    return datasets

def detect_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Automatically detects numerical and categorical features in a DataFrame."""
    # Columns with numeric data types are numerical
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    # All other columns are categorical
    categorical_features = df.select_dtypes(exclude=['number']).columns.tolist()
    
    logger.info(f"Detected {len(numeric_features)} numeric features: {numeric_features}")
    logger.info(f"Detected {len(categorical_features)} categorical features: {categorical_features}")
    
    return numeric_features, categorical_features

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer preprocessor pipeline."""
    logger.info("Building numerical and categorical pipelines...")
    
    # Numerical pipeline: Impute missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing values with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Set output to pandas DataFrame (automatically converts output matrices to dense DataFrames)
    preprocessor.set_output(transform="pandas")
    logger.info("Preprocessor pipeline successfully built.")
    return preprocessor

def log_missing_values(df: pd.DataFrame, dataset_name: str, phase: str) -> None:
    """Counts and logs missing values in a DataFrame."""
    missing = df.isnull().sum()
    total_missing = missing.sum()
    logger.info(f"Missing values in {dataset_name} ({phase} preprocessing) - Total Missing: {total_missing:,}")
    
    cols_with_missing = missing[missing > 0]
    if not cols_with_missing.empty:
        for col, count in cols_with_missing.items():
            logger.info(f"  - {col}: {count:,}")
    else:
        logger.info("  - No columns with missing values.")

def preprocess_inference_data(df: pd.DataFrame, preprocessor_path: str = "artifacts/preprocessor.joblib") -> pd.DataFrame:
    """
    Utility function to preprocess new raw data for model inference.
    Loads the saved preprocessor and transforms the data.
    """
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor object not found at: {preprocessor_path}")
        
    logger.info(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    
    # If target or split helper columns are present, drop them
    drop_cols = [col for col in ['loan_status', 'issue_year'] if col in df.columns]
    if drop_cols:
        logger.info(f"Dropping target/helper columns for inference: {drop_cols}")
        df = df.drop(columns=drop_cols)
        
    logger.info("Preprocessing inference data...")
    return preprocessor.transform(df)

def save_outputs(
    transformed_datasets: Dict[str, pd.DataFrame], 
    preprocessor: ColumnTransformer, 
    feature_names: List[str],
    output_dir: str, 
    artifacts_dir: str
) -> None:
    """Saves the preprocessed feature matrices, preprocessor pipeline object, and feature names list."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Save preprocessor object
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    logger.info(f"Saving fitted preprocessor pipeline to: {preprocessor_path}")
    joblib.dump(preprocessor, preprocessor_path)
    
    # 2. Save feature names
    feature_names_path = os.path.join(artifacts_dir, "feature_names.json")
    logger.info(f"Saving preprocessed feature names to: {feature_names_path}")
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=4)
        
    # 3. Save transformed datasets
    for split, df_processed in transformed_datasets.items():
        output_path = os.path.join(output_dir, f"X_{split}_processed.csv")
        logger.info(f"Saving transformed X_{split}_processed.csv ({df_processed.shape[0]} rows, {df_processed.shape[1]} features)...")
        df_processed.to_csv(output_path, index=False)

def main():
    start_time = time.time()
    
    splits_dir = os.path.join("data", "splits")
    output_dir = os.path.join("data", "processed_features")
    artifacts_dir = "artifacts"
    
    try:
        # Load datasets
        datasets = load_datasets(splits_dir)
        X_train = datasets["train"]
        
        # Log missing values BEFORE preprocessing
        print("\n" + "="*60)
        print("          MISSING VALUES REPORT (BEFORE PREPROCESSING)")
        print("="*60)
        for split, df in datasets.items():
            log_missing_values(df, f"X_{split}", "BEFORE")
        print("="*60 + "\n")
        
        # Detect feature types
        numeric_features, categorical_features = detect_feature_types(X_train)
        
        # Build preprocessor
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        
        # Fit preprocessor only on Train data
        logger.info("Fitting preprocessor pipeline on X_train...")
        preprocessor.fit(X_train)
        
        # Extract preprocessed feature names after fitting
        feature_names = preprocessor.get_feature_names_out().tolist()
        logger.info(f"Fitted preprocessor outputs {len(feature_names)} features (expanded from {X_train.shape[1]} original features).")
        
        # Transform datasets
        logger.info("Transforming train, val, test, and prod datasets...")
        transformed_datasets = {}
        for split, df in datasets.items():
            transformed_datasets[split] = preprocessor.transform(df)
            # Log missing values AFTER preprocessing to confirm imputation
            log_missing_values(transformed_datasets[split], f"X_{split}_processed", "AFTER")
            
        # Save preprocessor, feature names, and preprocessed datasets
        save_outputs(transformed_datasets, preprocessor, feature_names, output_dir, artifacts_dir)
        
        # Print Diagnostics & Reporting
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("           FEATURE PREPROCESSING DIAGNOSTIC REPORT")
        print("="*60)
        print(f"Original Feature Count      : {X_train.shape[1]}")
        print(f"Detected Numeric Features   : {len(numeric_features)}")
        print(f"Detected Categorical Features: {len(categorical_features)}")
        print(f"Output Feature Count        : {len(feature_names)}")
        print(f"\nTransformed Dataset Shapes:")
        for split, df in transformed_datasets.items():
            print(f"  - X_{split}_processed : {df.shape}")
        print(f"\nTotal Processing Time       : {total_time:.4f} seconds")
        print("="*60)
        
    except Exception as e:
        logger.exception("An error occurred during feature preprocessing:")
        sys.exit(1)

if __name__ == "__main__":
    main()
