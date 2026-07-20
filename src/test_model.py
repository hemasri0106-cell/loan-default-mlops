import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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

def load_registry(registry_path: Path) -> Dict[str, Any]:
    """Loads the model registry JSON to identify the selected champion model."""
    logger.info(f"Loading model registry from: {registry_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found at: {registry_path}")
    with open(registry_path, "r") as f:
        registry = json.load(f)
    return registry

def load_model(model_path: Path) -> Any:
    """Loads the serialized model joblib file."""
    logger.info(f"Loading champion model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def load_test_data(
    features_path: Path, 
    target_path: Path
) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads the processed test features and original target label split."""
    logger.info(f"Loading processed test features from: {features_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Test features not found at: {features_path}")
        
    logger.info(f"Loading test labels from: {target_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Test labels not found at: {target_path}")
        
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(target_path).squeeze("columns")
    
    # Ensure y_test is a pandas Series
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
        
    return X_test, y_test

def evaluate_metrics(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """Generates predictions and evaluates performance metrics, predictions df, and class report."""
    logger.info("Generating predictions on the test dataset...")
    y_pred = model.predict(X_test)
    
    # Generate prediction probabilities (class 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # fallback for linear classifiers if predict_proba is not directly available (rare for Sklearn/XGBoost classifiers here)
        y_prob = model.decision_function(X_test)
    else:
        logger.warning("Model does not support probability estimation. Using binary predictions for ROC-AUC.")
        y_prob = y_pred
        
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
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
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "Actual": y_test,
        "Prediction": y_pred,
        "Probability": y_prob
    })
    
    # Generate scikit-learn classification report
    cls_report = classification_report(y_test, y_pred, zero_division=0)
    
    logger.info("Test set performance evaluation completed successfully.")
    return metrics, predictions_df, cls_report

def save_results(
    metrics: Dict[str, Any], 
    predictions_df: pd.DataFrame, 
    cls_report: str, 
    results_dir: Path, 
    model_key: str
) -> None:
    """Saves predictions CSV, metrics JSON, and classification report TXT to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save predictions CSV
    predictions_path = results_dir / "test_predictions.csv"
    logger.info(f"Saving test predictions CSV to: {predictions_path}")
    predictions_df.to_csv(predictions_path, index=False)
    
    # 2. Save classification report TXT
    report_path = results_dir / "test_classification_report.txt"
    logger.info(f"Saving classification report text to: {report_path}")
    with open(report_path, "w") as f:
        f.write(cls_report)
        
    # 3. Save metrics JSON
    metrics_path = results_dir / "test_metrics.json"
    logger.info(f"Saving test metrics JSON to: {metrics_path}")
    
    output_metrics = {
        "model": model_key,
        "dataset": "test",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"]
    }
    
    with open(metrics_path, "w") as f:
        json.dump(output_metrics, f, indent=4)

def main():
    # Setup directories
    base_dir = Path("d:/onedrive/Desktop/loan predictor")
    registry_path = base_dir / "artifacts" / "model_registry.json"
    features_path = base_dir / "data" / "processed_features" / "X_test_processed.csv"
    target_path = base_dir / "data" / "splits" / "y_test.csv"
    results_dir = base_dir / "artifacts" / "results"
    
    try:
        # Load registry
        registry = load_registry(registry_path)
        selected_model = registry["selected_model"]
        relative_model_path = registry["model_path"]
        
        # Resolve full model file path
        model_file_path = base_dir / relative_model_path
        
        # Load model
        model = load_model(model_file_path)
        logger.info(f"Champion model '{selected_model}' successfully loaded.")
        
        # Load test datasets
        X_test, y_test = load_test_data(features_path, target_path)
        
        # Evaluate metrics
        metrics, predictions_df, cls_report = evaluate_metrics(model, X_test, y_test)
        
        # Save output results
        save_results(metrics, predictions_df, cls_report, results_dir, selected_model)
        
        # Print final console report
        print("\n" + "="*45)
        print("         Final Model Test Evaluation")
        print("="*45)
        print(f"Model:    {registry['selected_model'].replace('_', ' ').title()}")
        print("Dataset:  2017 Test Set")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision:{metrics['precision']:.4f}")
        print(f"Recall:   {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"  - TN: {cm['tn']:,} | FP: {cm['fp']:,}")
        print(f"  - FN: {cm['fn']:,} | TP: {cm['tp']:,}")
        
        print("\nClassification Report:")
        print(cls_report)
        
        print("Results saved to:")
        print(f"  - {results_dir / 'test_metrics.json'}")
        print(f"  - {results_dir / 'test_predictions.csv'}")
        print(f"  - {results_dir / 'test_classification_report.txt'}")
        print("="*45)
        
        logger.info("Model testing pipeline execution finished successfully.")
        
    except Exception as e:
        logger.exception("An error occurred during final model testing:")
        sys.exit(1)

if __name__ == "__main__":
    main()
