import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data(train_path: Path, prod_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the raw training and production feature datasets from CSV files."""
    logger.info(f"Loading raw training dataset from: {train_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Training features file not found at: {train_path}")
        
    logger.info(f"Loading raw production dataset from: {prod_path}")
    if not prod_path.exists():
        raise FileNotFoundError(f"Production features file not found at: {prod_path}")
        
    X_train = pd.read_csv(train_path)
    X_prod = pd.read_csv(prod_path)
    
    return X_train, X_prod

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Automatically detects numerical and categorical columns in a DataFrame."""
    logger.info("Detecting feature types automatically...")
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    logger.info(f"Detected {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features.")
    return numeric_cols, categorical_cols

def calculate_numeric_psi(train_series: pd.Series, prod_series: pd.Series) -> float:
    """Calculates Population Stability Index (PSI) for a numeric feature using 10 quantile bins."""
    train_clean = train_series.dropna()
    prod_clean = prod_series.dropna()
    
    if len(train_clean) == 0 or len(prod_clean) == 0:
        return 0.0
        
    # Create 10 quantile bins based on the training data distribution
    percentiles = np.linspace(0, 100, 11)
    bin_edges = np.percentile(train_clean, percentiles)
    
    # Make bin edges unique to avoid interval mapping errors for low-variance features
    bin_edges = np.unique(bin_edges)
    
    # Handle the edge case of all identical training values
    if len(bin_edges) == 1:
        bin_edges = np.array([bin_edges[0] - 0.001, bin_edges[0] + 0.001])
    else:
        # Expand bounds to encompass all train and prod values
        bin_edges[0] = min(bin_edges[0], train_clean.min(), prod_clean.min()) - 1e-5
        bin_edges[-1] = max(bin_edges[-1], train_clean.max(), prod_clean.max()) + 1e-5
        
    # Discretize datasets using histogram
    train_counts, _ = np.histogram(train_clean, bins=bin_edges)
    prod_counts, _ = np.histogram(prod_clean, bins=bin_edges)
    
    # Calculate initial proportions
    train_props = train_counts / len(train_clean)
    prod_props = prod_counts / len(prod_clean)
    
    # Handle zero-frequency bins safely by adding a small epsilon
    epsilon = 1e-4
    train_props = np.where(train_props == 0, epsilon, train_props)
    prod_props = np.where(prod_props == 0, epsilon, prod_props)
    
    # Re-normalize proportions so they sum to 1.0 after epsilon correction
    train_props /= train_props.sum()
    prod_props /= prod_props.sum()
    
    # Calculate PSI: sum( (Actual - Expected) * ln(Actual / Expected) )
    psi_value = np.sum((prod_props - train_props) * np.log(prod_props / train_props))
    return float(psi_value)

def calculate_categorical_drift(train_series: pd.Series, prod_series: pd.Series) -> float:
    """Calculates total absolute difference in class frequencies for a categorical feature."""
    train_clean = train_series.dropna().astype(str)
    prod_clean = prod_series.dropna().astype(str)
    
    if len(train_clean) == 0 or len(prod_clean) == 0:
        return 0.0
        
    train_freqs = train_clean.value_counts(normalize=True)
    prod_freqs = prod_clean.value_counts(normalize=True)
    
    # Get the union of all categorical categories present in both datasets
    all_categories = set(train_freqs.index).union(set(prod_freqs.index))
    
    # Calculate total absolute difference: sum(|Train Freq - Prod Freq|)
    total_diff = 0.0
    for cat in all_categories:
        f_train = train_freqs.get(cat, 0.0)
        f_prod = prod_freqs.get(cat, 0.0)
        total_diff += abs(f_train - f_prod)
        
    return float(total_diff)

def get_drift_status(value: float) -> str:
    """Interprets the drift metric to determine status."""
    if value < 0.10:
        return "No Drift"
    elif value < 0.25:
        return "Moderate Drift"
    else:
        return "Significant Drift"

def calculate_drift_for_all(
    X_train: pd.DataFrame, 
    X_prod: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> List[Dict[str, Any]]:
    """Computes data drift statistics for all columns in the datasets."""
    logger.info("Computing drift metrics for all features...")
    results = []
    
    # Compute numeric features drift (PSI)
    for col in numeric_cols:
        psi = calculate_numeric_psi(X_train[col], X_prod[col])
        status = get_drift_status(psi)
        results.append({
            "Feature": col,
            "Feature_Type": "Numerical",
            "Metric": psi,
            "Drift_Status": status
        })
        
    # Compute categorical features drift (Total Frequency Absolute Difference)
    for col in categorical_cols:
        diff = calculate_categorical_drift(X_train[col], X_prod[col])
        status = get_drift_status(diff)
        results.append({
            "Feature": col,
            "Feature_Type": "Categorical",
            "Metric": diff,
            "Drift_Status": status
        })
        
    return results

def save_detailed_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Saves a detailed drift CSV report sorted by metric descending."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving detailed drift report CSV to: {output_path}")
    
    df = pd.DataFrame(results)
    # Sort descending by Metric (PSI / total difference)
    df = df.sort_values(by="Metric", ascending=False).reset_index(drop=True)
    df.to_csv(output_path, index=False)

def save_drift_summary(
    results: List[Dict[str, Any]], 
    output_path: Path
) -> Dict[str, Any]:
    """Creates, saves, and returns the final drift summary JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving drift summary JSON to: {output_path}")
    
    total_features = len(results)
    
    # Count drift statuses
    no_drift = sum(1 for r in results if r["Drift_Status"] == "No Drift")
    moderate_drift = sum(1 for r in results if r["Drift_Status"] == "Moderate Drift")
    significant_drift = sum(1 for r in results if r["Drift_Status"] == "Significant Drift")
    
    # Decision Rules:
    # 1. Any feature has Significant Drift (PSI / Difference >= 0.25)
    has_significant_drift = significant_drift > 0
    
    # 2. More than 30% of all features show at least moderate drift
    moderate_or_above_ratio = (moderate_drift + significant_drift) / total_features
    high_moderate_drift_percentage = moderate_or_above_ratio > 0.30
    
    # Determine retraining recommendation
    if has_significant_drift or high_moderate_drift_percentage:
        overall_status = "Retraining Recommended"
        retraining_recommended = True
    else:
        overall_status = "No Retraining Needed"
        retraining_recommended = False
        
    summary_data = {
        "training_dataset": "2007-2015",
        "production_dataset": "2018",
        "total_features": total_features,
        "no_drift": no_drift,
        "moderate_drift": moderate_drift,
        "significant_drift": significant_drift,
        "overall_status": overall_status,
        "retraining_recommended": retraining_recommended
    }
    
    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    return summary_data

def main():
    # Setup directories
    base_dir = Path("d:/onedrive/Desktop/loan predictor")
    train_path = base_dir / "data" / "splits" / "X_train.csv"
    prod_path = base_dir / "data" / "splits" / "X_prod.csv"
    
    artifacts_dir = base_dir / "artifacts" / "drift"
    report_path = artifacts_dir / "drift_report.csv"
    summary_path = artifacts_dir / "drift_summary.json"
    
    try:
        # Load datasets
        X_train, X_prod = load_data(train_path, prod_path)
        logger.info("Training and production datasets loaded successfully.")
        
        # Detect column types
        numeric_cols, categorical_cols = detect_column_types(X_train)
        
        # Calculate drift
        results = calculate_drift_for_all(X_train, X_prod, numeric_cols, categorical_cols)
        logger.info("Drift calculations completed.")
        
        # Save CSV report
        save_detailed_report(results, report_path)
        
        # Save JSON summary
        summary = save_drift_summary(results, summary_path)
        
        # Print final console report
        print("\n" + "="*45)
        print("             Data Drift Detection")
        print("="*45)
        print("Training Dataset:   2007–2015")
        print("Production Dataset: 2018")
        print(f"Total Features:     {summary['total_features']}")
        print("-" * 45)
        print(f"No Drift:           {summary['no_drift']}")
        print(f"Moderate Drift:     {summary['moderate_drift']}")
        print(f"Significant Drift:  {summary['significant_drift']}")
        print("-" * 45)
        print("Overall Decision:")
        print(f"  ** {summary['overall_status']} **")
        print("\nArtifacts Saved:")
        print(f"  - {report_path}")
        print(f"  - {summary_path}")
        print("="*45)
        
        logger.info("Drift detection pipeline execution finished successfully.")
        
    except Exception as e:
        logger.exception("An error occurred during drift detection:")
        sys.exit(1)

if __name__ == "__main__":
    main()
