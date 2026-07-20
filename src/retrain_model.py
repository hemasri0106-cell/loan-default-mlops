import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def detect_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Automatically detects numerical and categorical features in a DataFrame."""
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_features, categorical_features

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer preprocessor pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor

def load_json(path: Path) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def build_champion_model(selected_model_key: str, y_train_combined: pd.Series) -> Any:
    """Instantiates the champion model with identical hyperparameters from original training."""
    logger.info(f"Instantiating model algorithm for: {selected_model_key}")
    key = selected_model_key.lower()
    
    if "logistic" in key:
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )
    elif "random" in key or "forest" in key:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif "xgboost" in key:
        neg_samples = int((y_train_combined == 0).sum())
        pos_samples = int((y_train_combined == 1).sum())
        scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
        logger.info(f"Calculated scale_pos_weight for combined training set: {scale_pos_weight:.6f}")
        return XGBClassifier(
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
    else:
        raise ValueError(f"Unsupported model key in registry: {selected_model_key}")

def evaluate_retrained_model(model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
    """Evaluates the retrained model on the validation dataset."""
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = y_pred

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    return {
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

def main():
    base_dir = Path("d:/onedrive/Desktop/loan predictor")
    artifacts_dir = base_dir / "artifacts"
    drift_summary_path = artifacts_dir / "drift" / "drift_summary.json"
    registry_path = artifacts_dir / "model_registry.json"
    test_metrics_path = artifacts_dir / "results" / "test_metrics.json"
    results_dir = artifacts_dir / "results"
    splits_dir = base_dir / "data" / "splits"

    logger.info("Starting Automated Retraining Pipeline...")

    # Step 1: Check Drift Summary
    drift_summary = load_json(drift_summary_path)
    retraining_recommended = drift_summary.get("retraining_recommended", False)
    
    registry = load_json(registry_path)
    selected_model_key = registry.get("selected_model", "logistic_regression")
    model_name_display = selected_model_key.replace("_", " ").title()

    if not retraining_recommended:
        logger.info("No data drift detected requiring retraining.")
        print("\n" + "="*45)
        print("      Automated Retraining Pipeline")
        print("="*45)
        print(f"Champion Model:  {model_name_display}")
        print("Drift Detected:  NO")
        print("-" * 45)
        print("Decision:        No retraining required.")
        print("Status:          Champion model remains deployed.")
        print("="*45)
        sys.exit(0)

    logger.info("Data drift detected! Retraining recommended. Proceeding with retraining pipeline...")

    # Step 2: Load Data & Combine
    logger.info("Loading training (2007-2015), production (2018), and validation (2016) splits...")
    X_train = pd.read_csv(splits_dir / "X_train.csv")
    y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze("columns")
    
    X_prod = pd.read_csv(splits_dir / "X_prod.csv")
    y_prod = pd.read_csv(splits_dir / "y_prod.csv").squeeze("columns")

    X_val = pd.read_csv(splits_dir / "X_val.csv")
    y_val = pd.read_csv(splits_dir / "y_val.csv").squeeze("columns")

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_prod, pd.DataFrame):
        y_prod = y_prod.iloc[:, 0]
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.iloc[:, 0]

    # Combine historical train (2007-2015) + production (2018)
    X_train_combined = pd.concat([X_train, X_prod], ignore_index=True)
    y_train_combined = pd.concat([y_train, y_prod], ignore_index=True)
    logger.info(f"Combined training dataset shape: {X_train_combined.shape} (rows: {len(X_train_combined):,})")

    # Step 3: Rebuild Preprocessing Pipeline
    logger.info("Fitting a brand-new preprocessing pipeline on combined training data...")
    numeric_features, categorical_features = detect_feature_types(X_train_combined)
    preprocessor_latest = build_preprocessor(numeric_features, categorical_features)
    
    X_train_combined_processed = preprocessor_latest.fit_transform(X_train_combined)
    X_val_processed = preprocessor_latest.transform(X_val)

    # Save latest preprocessor
    latest_preprocessor_path = artifacts_dir / "preprocessor_latest.joblib"
    logger.info(f"Saving latest preprocessor to: {latest_preprocessor_path}")
    joblib.dump(preprocessor_latest, latest_preprocessor_path)

    # Step 4: Retrain Champion Model
    model = build_champion_model(selected_model_key, y_train_combined)

    # Handle XGBoost feature name cleaning if necessary
    if "xgboost" in selected_model_key.lower():
        X_train_combined_processed.columns = [col.replace('<', 'less_than').replace('[', '(').replace(']', ')') for col in X_train_combined_processed.columns]
        X_val_processed.columns = [col.replace('<', 'less_than').replace('[', '(').replace(']', ')') for col in X_val_processed.columns]

    logger.info(f"Fitting {model_name_display} on combined dataset...")
    start_time = time.time()
    model.fit(X_train_combined_processed, y_train_combined)
    training_time = time.time() - start_time
    logger.info(f"Retraining completed in {training_time:.4f} seconds.")

    # Step 5: Evaluate Retrained Model on Validation Split
    logger.info("Evaluating retrained model on validation dataset...")
    retrained_metrics = evaluate_retrained_model(model, X_val_processed, y_val)
    retrained_metrics["training_time_seconds"] = training_time

    # Save retrained metrics JSON
    retrained_metrics_path = results_dir / "retrained_metrics.json"
    logger.info(f"Saving retrained metrics JSON to: {retrained_metrics_path}")
    with open(retrained_metrics_path, "w") as f:
        json.dump({
            "model": selected_model_key,
            "dataset": "validation_retrained",
            "training_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_time_seconds": training_time,
            "metrics": {
                "accuracy": retrained_metrics["accuracy"],
                "precision": retrained_metrics["precision"],
                "recall": retrained_metrics["recall"],
                "f1_score": retrained_metrics["f1_score"],
                "roc_auc": retrained_metrics["roc_auc"]
            },
            "confusion_matrix": retrained_metrics["confusion_matrix"]
        }, f, indent=4)

    # Step 6: Promotion Decision against Deployed Model Test Metrics
    test_metrics = load_json(test_metrics_path)
    deployed_roc_auc = test_metrics.get("roc_auc", 0.0)
    deployed_f1 = test_metrics.get("f1_score", 0.0)

    retrained_roc_auc = retrained_metrics["roc_auc"]
    retrained_f1 = retrained_metrics["f1_score"]

    logger.info(f"Deployed Test ROC-AUC: {deployed_roc_auc:.4f} | Deployed Test F1: {deployed_f1:.4f}")
    logger.info(f"Retrained Val ROC-AUC:  {retrained_roc_auc:.4f} | Retrained Val F1:  {retrained_f1:.4f}")

    # Promotion rules: ROC-AUC >= deployed ROC-AUC AND F1 Score does not drop by more than 1%
    roc_pass = retrained_roc_auc >= deployed_roc_auc
    f1_pass = retrained_f1 >= (deployed_f1 - 0.01)
    promoted = roc_pass and f1_pass

    # Step 7: Promotion Action
    if promoted:
        logger.info("Retrained model satisfied promotion criteria! Promoting new model...")
        promoted_model_path = artifacts_dir / "models" / f"{selected_model_key}_latest.joblib"
        logger.info(f"Saving promoted model object to: {promoted_model_path}")
        joblib.dump(model, promoted_model_path)

        current_version = registry.get("version", 1)
        new_version = current_version + 1
        
        promotion_reason = (
            f"Retrained model satisfied promotion criteria: ROC-AUC ({retrained_roc_auc:.4f}) >= "
            f"Deployed Test ROC-AUC ({deployed_roc_auc:.4f}), and F1 Score ({retrained_f1:.4f}) "
            f"did not decrease by more than 1% compared to deployed F1 ({deployed_f1:.4f})."
        )

        updated_registry = {
            "selected_model": selected_model_key,
            "model_path": f"artifacts/models/{selected_model_key}_latest.joblib",
            "version": new_version,
            "promotion_reason": promotion_reason,
            "previous_version": current_version,
            "promotion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "accuracy": retrained_metrics["accuracy"],
                "precision": retrained_metrics["precision"],
                "recall": retrained_metrics["recall"],
                "f1_score": retrained_metrics["f1_score"],
                "roc_auc": retrained_metrics["roc_auc"]
            }
        }

        with open(registry_path, "w") as f:
            json.dump(updated_registry, f, indent=4)
        logger.info("Model registry successfully updated.")
        promotion_status_text = "Accepted"
        registry_status_text = f"Updated (Version {new_version})"
    else:
        logger.info("Retrained model did not satisfy promotion criteria. Keeping current champion model.")
        promotion_status_text = "Rejected"
        registry_status_text = "Unchanged (Registry kept version 1)"

    # Print Console Summary
    print("\n" + "="*45)
    print("      Automated Retraining Pipeline")
    print("="*45)
    print(f"Champion Model:     {model_name_display}")
    print("Drift Detected:     YES")
    print("Retraining Status:  Completed")
    print("-" * 45)
    print(f"Retrained ROC-AUC:  {retrained_roc_auc:.4f}")
    print(f"Deployed Test AUC:  {deployed_roc_auc:.4f}")
    print(f"Retrained F1 Score: {retrained_f1:.4f}")
    print(f"Deployed Test F1:   {deployed_f1:.4f}")
    print("-" * 45)
    print(f"Promotion Decision: {promotion_status_text}")
    print(f"Registry Status:    {registry_status_text}")
    print("Artifacts Saved:")
    print(f"  - {latest_preprocessor_path}")
    print(f"  - {retrained_metrics_path}")
    if promoted:
        print(f"  - {artifacts_dir / 'models' / f'{selected_model_key}_latest.joblib'}")
        print(f"  - {registry_path}")
    print("="*45)

if __name__ == "__main__":
    main()
