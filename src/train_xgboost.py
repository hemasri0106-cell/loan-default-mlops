import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, List
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data(
    features_dir: Path, 
    splits_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads processed features and target split datasets from CSV files."""
    logger.info("Loading processed datasets...")
    
    # Define paths using Pathlib
    X_train_path = features_dir / "X_train_processed.csv"
    X_val_path = features_dir / "X_val_processed.csv"
    y_train_path = splits_dir / "y_train.csv"
    y_val_path = splits_dir / "y_val.csv"
    
    # Load feature sets
    X_train = pd.read_csv(X_train_path)
    X_val = pd.read_csv(X_val_path)
    
    # Load target sets and squeeze to Series
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_val = pd.read_csv(y_val_path).squeeze("columns")
    
    # Ensure they are indeed pandas Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.iloc[:, 0]
        
    logger.info("Successfully loaded all datasets.")
    return X_train, X_val, y_train, y_val

def validate_data(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    y_train: pd.Series, 
    y_val: pd.Series
) -> None:
    """Validates shapes, alignment, and missing value presence in the datasets."""
    logger.info("Validating dataset shapes and consistency...")
    
    # Log shapes
    logger.info(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    logger.info(f"X_val shape:   {X_val.shape} | y_val shape:   {y_val.shape}")
    
    # 1. Row counts match check
    if len(X_train) != len(y_train):
        raise ValueError(f"Train row mismatch! X_train has {len(X_train)} rows but y_train has {len(y_train)} rows.")
    if len(X_val) != len(y_val):
        raise ValueError(f"Validation row mismatch! X_val has {len(X_val)} rows but y_val has {len(y_val)} rows.")
        
    # 2. Missing values check
    train_features_nan = X_train.isnull().sum().sum()
    train_target_nan = y_train.isnull().sum()
    val_features_nan = X_val.isnull().sum().sum()
    val_target_nan = y_val.isnull().sum()
    
    if train_features_nan > 0 or train_target_nan > 0:
        raise ValueError(
            f"Validation failed: Missing values found in training split! "
            f"(X_train NaNs: {train_features_nan}, y_train NaNs: {train_target_nan})"
        )
    if val_features_nan > 0 or val_target_nan > 0:
        raise ValueError(
            f"Validation failed: Missing values found in validation split! "
            f"(X_val NaNs: {val_features_nan}, y_val NaNs: {val_target_nan})"
        )
        
    # 3. Log target class distribution
    train_counts = y_train.value_counts()
    train_props = y_train.value_counts(normalize=True) * 100
    
    logger.info("Training target class distribution:")
    for cls in sorted(train_counts.keys()):
        logger.info(f"  - Class {cls}: {train_counts[cls]:,} ({train_props[cls]:.2f}%)")
        
    logger.info("Dataset validation checks passed successfully.")

def build_model(scale_pos_weight: float) -> XGBClassifier:
    """Builds the baseline XGBoost model."""
    logger.info(f"Building XGBoost model with scale_pos_weight={scale_pos_weight:.6f}...")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    return model

def train_model(
    model: XGBClassifier, 
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Tuple[XGBClassifier, float]:
    """Trains the baseline model and measures execution duration."""
    logger.info("Starting baseline model training (XGBoost)...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Model training completed in {elapsed_time:.4f} seconds.")
    return model, elapsed_time

def extract_feature_importance(model: XGBClassifier, feature_names: List[str]) -> pd.DataFrame:
    """Extracts feature importances and maps them to feature names, sorted descending."""
    logger.info("Extracting feature importances...")
    importances = model.feature_importances_
    
    importances_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    return importances_df

def evaluate_model(
    model: XGBClassifier, 
    X_val: pd.DataFrame, 
    y_val: pd.Series
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates the model on validation features and returns the evaluation metrics & prediction details."""
    logger.info("Running validation predictions and metric evaluation...")
    
    # Predictions
    y_pred = model.predict(X_val)
    # Predict probability of the positive class (class 1)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    
    # Create prediction details DataFrame with required columns: actual, predicted, probability_default
    predictions_df = pd.DataFrame({
        "actual": y_val,
        "predicted": y_pred,
        "probability_default": y_prob
    })
    
    logger.info("Validation metrics calculation complete.")
    return metrics, predictions_df

def save_model(model: XGBClassifier, model_dir: Path) -> None:
    """Saves the trained model object using joblib."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost.joblib"
    logger.info(f"Saving trained model object to: {model_path}")
    joblib.dump(model, model_path)

def save_predictions(predictions_df: pd.DataFrame, results_dir: Path) -> None:
    """Saves validation predictions to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / "xgboost_val_predictions.csv"
    logger.info(f"Saving validation predictions CSV to: {predictions_path}")
    predictions_df.to_csv(predictions_path, index=False)

def save_metrics(
    metrics: Dict[str, Any], 
    importances_df: pd.DataFrame,
    results_dir: Path, 
    training_time: float,
    scale_pos_weight: float
) -> None:
    """Saves feature importances CSV and JSON report metrics to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save feature importances CSV
    importances_path = results_dir / "xgboost_feature_importance.csv"
    logger.info(f"Saving feature importances CSV to: {importances_path}")
    importances_df.to_csv(importances_path, index=False)
    
    # 2. Build and save JSON report
    metrics_path = results_dir / "xgboost_metrics.json"
    logger.info(f"Saving metadata metrics JSON report to: {metrics_path}")
    
    report_data = {
        "model_name": "XGBoost Baseline",
        "algorithm": "XGBClassifier",
        "training_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_used": "Lending Club Preprocessed Splits",
        "training_time_seconds": training_time,
        "scale_pos_weight": scale_pos_weight,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        },
        "confusion_matrix": metrics["confusion_matrix"]
    }
    
    with open(metrics_path, "w") as f:
        json.dump(report_data, f, indent=4)

def main():
    # Setup directories
    base_dir = Path("d:/onedrive/Desktop/loan predictor")
    features_dir = base_dir / "data" / "processed_features"
    splits_dir = base_dir / "data" / "splits"
    
    artifacts_dir = base_dir / "artifacts"
    model_dir = artifacts_dir / "models"
    results_dir = artifacts_dir / "results"
    
    try:
        # Load data
        X_train, X_val, y_train, y_val = load_data(features_dir, splits_dir)
        
        # Clean feature names to remove [, ] or < which are forbidden by XGBoost
        X_train.columns = [col.replace('<', 'less_than').replace('[', '(').replace(']', ')') for col in X_train.columns]
        X_val.columns = [col.replace('<', 'less_than').replace('[', '(').replace(']', ')') for col in X_val.columns]
        
        # Validate data & log distributions
        validate_data(X_train, X_val, y_train, y_val)
        
        # Calculate scale_pos_weight: negative samples / positive samples
        neg_samples = int((y_train == 0).sum())
        pos_samples = int((y_train == 1).sum())
        scale_pos_weight = neg_samples / pos_samples
        logger.info(f"Calculated scale_pos_weight (neg_samples/pos_samples): {scale_pos_weight:.6f}")
        
        # Build model
        model = build_model(scale_pos_weight)
        
        # Train model
        model, training_time = train_model(model, X_train, y_train)
        
        # Extract feature importances mapping to preprocessed features columns
        feature_names = X_train.columns.tolist()
        importances_df = extract_feature_importance(model, feature_names)
        
        # Evaluate model
        metrics, predictions_df = evaluate_model(model, X_val, y_val)
        
        # Save model
        save_model(model, model_dir)
        
        # Save predictions CSV
        save_predictions(predictions_df, results_dir)
        
        # Save feature importances CSV and metrics JSON
        save_metrics(metrics, importances_df, results_dir, training_time, scale_pos_weight)
        
        # Print final console report
        print("\n" + "="*45)
        print("             XGBoost Results")
        print("="*45)
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1_score']:.4f}")
        print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  - TN: {cm['tn']:,} | FP: {cm['fp']:,}")
        print(f"  - FN: {cm['fn']:,} | TP: {cm['tp']:,}")
        
        print("\nTop 15 Feature Importances:")
        for idx, row in importances_df.head(15).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<35}: {row['importance']:.6f}")
            
        print("\nModel saved to:")
        print(f"  - {model_dir / 'xgboost.joblib'}")
        print("\nMetrics saved to:")
        print(f"  - {results_dir / 'xgboost_metrics.json'}")
        print("Feature importances saved to:")
        print(f"  - {results_dir / 'xgboost_feature_importance.csv'}")
        print("Validation predictions saved to:")
        print(f"  - {results_dir / 'xgboost_val_predictions.csv'}")
        print("="*45)
        
        logger.info("XGBoost training pipeline execution finished.")
        
    except Exception as e:
        logger.exception("An error occurred during model training:")
        sys.exit(1)

if __name__ == "__main__":
    main()
