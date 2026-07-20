import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class LoanPredictorService:
    _instance = None

    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path("d:/onedrive/Desktop/loan predictor")
        self.base_dir = base_dir
        self.registry_path = self.base_dir / "artifacts" / "model_registry.json"
        
        self.model = None
        self.preprocessor = None
        self.registry_info = {}
        
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads model registry, champion model, and latest preprocessor once at startup."""
        logger.info("Initializing Loan Predictor Service artifacts...")
        
        # 1. Load Registry Metadata
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Model registry not found at: {self.registry_path}")
            
        with open(self.registry_path, "r") as f:
            self.registry_info = json.load(f)
            
        relative_model_path = self.registry_info.get("model_path", "artifacts/models/logistic_regression.joblib")
        model_full_path = self.base_dir / relative_model_path
        
        # Fallback if specific version file is absent
        if not model_full_path.exists():
            fallback_model = self.base_dir / "artifacts" / "models" / "logistic_regression.joblib"
            logger.warning(f"Model file {model_full_path} not found. Falling back to: {fallback_model}")
            model_full_path = fallback_model
            
        logger.info(f"Loading champion model from: {model_full_path}")
        self.model = joblib.load(model_full_path)

        # 2. Load Preprocessor (Prefer preprocessor_latest.joblib, fallback to preprocessor.joblib)
        latest_preprocessor = self.base_dir / "artifacts" / "preprocessor_latest.joblib"
        original_preprocessor = self.base_dir / "artifacts" / "preprocessor.joblib"
        
        if latest_preprocessor.exists():
            preprocessor_path = latest_preprocessor
            logger.info(f"Loading latest preprocessor from: {preprocessor_path}")
        elif original_preprocessor.exists():
            preprocessor_path = original_preprocessor
            logger.info(f"Loading baseline preprocessor from: {preprocessor_path}")
        else:
            raise FileNotFoundError("Neither preprocessor_latest.joblib nor preprocessor.joblib found in artifacts!")
            
        self.preprocessor = joblib.load(preprocessor_path)
        logger.info("All model artifacts successfully loaded into memory.")

    def get_model_info(self) -> Dict[str, Any]:
        """Returns the current model registry metadata."""
        return {
            "selected_model": self.registry_info.get("selected_model", "Unknown").replace("_", " ").title(),
            "model_key": self.registry_info.get("selected_model", "logistic_regression"),
            "model_path": self.registry_info.get("model_path", "N/A"),
            "version": self.registry_info.get("version", 1),
            "promotion_reason": self.registry_info.get("promotion_reason", self.registry_info.get("selection_reason", "Champion model selected.")),
            "promotion_date": self.registry_info.get("promotion_date", self.registry_info.get("selection_date", "N/A")),
            "previous_version": self.registry_info.get("previous_version", "None"),
            "metrics": self.registry_info.get("metrics", {})
        }

    def predict(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Runs preprocessor transformation and model inference on input features."""
        start_time = time.time()
        
        # Coerce numeric types
        numeric_cols = [
            "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
            "fico_range_low", "fico_range_high", "open_acc", "revol_util"
        ]
        
        cleaned_input = {}
        for k, v in raw_features.items():
            if k in numeric_cols:
                try:
                    cleaned_input[k] = float(v)
                except (ValueError, TypeError):
                    cleaned_input[k] = 0.0
            else:
                cleaned_input[k] = str(v)
                
        # Create 1-row DataFrame
        df = pd.DataFrame([cleaned_input])
        
        # Transform features
        X_processed = self.preprocessor.transform(df)
        
        # Check XGBoost column compatibility if necessary
        model_type = self.registry_info.get("selected_model", "").lower()
        if "xgboost" in model_type:
            X_processed.columns = [
                col.replace('<', 'less_than').replace('[', '(').replace(']', ')')
                for col in X_processed.columns
            ]
            
        # Predict
        y_pred = self.model.predict(X_processed)
        
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_processed)
            prob_default = float(y_prob[0][1])
        else:
            prob_default = float(y_pred[0])
            
        pred_val = int(y_pred[0])
        
        if pred_val == 0:
            prediction_text = "Loan Likely to be Repaid"
            prediction_label = "Low Risk"
        else:
            prediction_text = "High Default Risk"
            prediction_label = "High Risk"
            
        risk_score = int(round(prob_default * 100))
        latency = round((time.time() - start_time) * 1000, 2)
        
        return {
            "prediction": prediction_text,
            "prediction_label": prediction_label,
            "probability": round(prob_default, 4),
            "risk_score": risk_score,
            "latency_ms": latency,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Singleton accessor
_service_instance = None

def get_predictor_service() -> LoanPredictorService:
    global _service_instance
    if _service_instance is None:
        _service_instance = LoanPredictorService()
    return _service_instance
