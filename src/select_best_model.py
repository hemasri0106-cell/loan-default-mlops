import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from functools import cmp_to_key
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

def load_model_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """Loads all model metric JSON files from the results directory, ignoring archived ones."""
    logger.info("Scanning results directory for metric JSON files...")
    models_metrics = []
    
    # Locate all JSON files in the results directory
    for file_path in results_dir.glob("*.json"):
        # Ignore archived v1 results and the model registry itself
        if file_path.name.endswith("_v1.json") or file_path.name == "model_registry.json" or "comparison" in file_path.name:
            continue
            
        logger.info(f"Loading metrics from: {file_path}")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            # Perform basic key validation
            if "model_name" in data and "metrics" in data:
                models_metrics.append({
                    "file_name": file_path.name,
                    "model_name": data["model_name"],
                    "accuracy": data["metrics"]["accuracy"],
                    "precision": data["metrics"]["precision"],
                    "recall": data["metrics"]["recall"],
                    "f1_score": data["metrics"]["f1_score"],
                    "roc_auc": data["metrics"]["roc_auc"],
                    "training_time": data.get("training_time_seconds", 0.0),
                    "raw_data": data
                })
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            
    logger.info(f"Successfully loaded {len(models_metrics)} model metric reports.")
    return models_metrics

def compare_models(m1: Dict[str, Any], m2: Dict[str, Any]) -> int:
    """Comparator to rank models based on the priority rules.
    Returns:
       -1 if m1 is better than m2 (m1 should rank higher)
        1 if m2 is better than m1 (m2 should rank higher)
        0 if they are tied
    """
    roc_diff = m1["roc_auc"] - m2["roc_auc"]
    
    # If the absolute difference in ROC-AUC is >= 0.005, the one with higher ROC-AUC is better
    if abs(roc_diff) >= 0.005:
        return -1 if roc_diff > 0 else 1
        
    # If the difference is < 0.005, treat them as statistically similar on ROC-AUC
    # and break the tie using F1 Score
    f1_diff = m1["f1_score"] - m2["f1_score"]
    if abs(f1_diff) > 1e-6:
        return -1 if f1_diff > 0 else 1
        
    # If F1 Score is also tied, break the tie using Recall
    rec_diff = m1["recall"] - m2["recall"]
    if abs(rec_diff) > 1e-6:
        return -1 if rec_diff > 0 else 1
        
    # If Recall is also tied, break the tie using Training Time (lower is better)
    time_diff = m1["training_time"] - m2["training_time"]
    if abs(time_diff) > 1e-6:
        return -1 if time_diff < 0 else 1
        
    return 0

def get_model_key_and_path(model_name: str) -> Tuple[str, str]:
    """Maps the model name to a registry-friendly key and standard artifact joblib path."""
    name_lower = model_name.lower()
    if "logistic" in name_lower:
        return "logistic_regression", "artifacts/models/logistic_regression.joblib"
    elif "random forest" in name_lower or "forest" in name_lower:
        return "random_forest", "artifacts/models/random_forest.joblib"
    elif "xgboost" in name_lower:
        return "xgboost", "artifacts/models/xgboost.joblib"
    else:
        clean_name = name_lower.replace(" ", "_").replace("baseline", "").strip("_")
        return clean_name, f"artifacts/models/{clean_name}.joblib"

def generate_selection_reason(selected: Dict[str, Any], runner_up: Dict[str, Any] = None) -> str:
    """Generates a text description justifying the selection of the best model."""
    if runner_up is None:
        return f"Selected {selected['model_name']} as the best model with ROC-AUC={selected['roc_auc']:.4f}."
        
    roc_diff = selected["roc_auc"] - runner_up["roc_auc"]
    
    # Case 1: Winner won by clean ROC-AUC gap
    if roc_diff >= 0.005:
        return (
            f"Highest ROC-AUC ({selected['roc_auc']:.4f}) on the validation split. "
            f"It outperforms the next best model ({runner_up['model_name']} with ROC-AUC={runner_up['roc_auc']:.4f}) "
            f"by a significant margin of {roc_diff:.4f}."
        )
        
    # Case 2: Winner won by tie-breaker because ROC-AUC was statistically similar
    reason = (
        f"Statistically similar ROC-AUC to {runner_up['model_name']} (difference of {abs(roc_diff):.4f} is less than 0.005). "
        f"Selected as the best model because of a higher F1 Score ({selected['f1_score']:.4f} vs {runner_up['f1_score']:.4f}) "
    )
    if selected["recall"] > runner_up["recall"]:
        reason += f"and higher Recall ({selected['recall']:.4f} vs {runner_up['recall']:.4f})."
    else:
        reason += f"and competitive Recall."
        
    return reason

def save_comparison_csv(ranked_models: List[Dict[str, Any]], output_path: Path) -> None:
    """Saves the comparison dataframe to a CSV file."""
    logger.info(f"Saving comparison CSV to: {output_path}")
    
    rows = []
    for rank, model in enumerate(ranked_models, 1):
        rows.append({
            "Model": model["model_name"],
            "Accuracy": model["accuracy"],
            "Precision": model["precision"],
            "Recall": model["recall"],
            "F1 Score": model["f1_score"],
            "ROC-AUC": model["roc_auc"],
            "Training Time": model["training_time"],
            "Rank": rank
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

def save_model_registry(
    selected: Dict[str, Any], 
    registry_path: Path, 
    reason: str
) -> None:
    """Creates and saves the model registry JSON file."""
    logger.info(f"Saving model registry to: {registry_path}")
    
    model_key, model_path = get_model_key_and_path(selected["model_name"])
    
    registry_data = {
        "selected_model": model_key,
        "selection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_reason": reason,
        "metrics": {
            "accuracy": selected["accuracy"],
            "precision": selected["precision"],
            "recall": selected["recall"],
            "f1_score": selected["f1_score"],
            "roc_auc": selected["roc_auc"],
            "training_time_seconds": selected["training_time"]
        },
        "model_path": model_path
    }
    
    with open(registry_path, "w") as f:
        json.dump(registry_data, f, indent=4)

def main():
    # Setup directories
    base_dir = Path("d:/onedrive/Desktop/loan predictor")
    results_dir = base_dir / "artifacts" / "results"
    artifacts_dir = base_dir / "artifacts"
    
    try:
        # Load metrics
        models_metrics = load_model_metrics(results_dir)
        
        if not models_metrics:
            logger.error("No valid model metrics found in the results directory!")
            sys.exit(1)
            
        # Rank models using the custom comparator
        ranked_models = sorted(models_metrics, key=cmp_to_key(compare_models))
        logger.info("Model ranking comparison complete.")
        
        # Identify the selected model and runner-up for reasoning text
        selected = ranked_models[0]
        runner_up = ranked_models[1] if len(ranked_models) > 1 else None
        reason = generate_selection_reason(selected, runner_up)
        
        # Save comparison CSV
        csv_path = results_dir / "model_comparison.csv"
        save_comparison_csv(ranked_models, csv_path)
        
        # Save model registry
        registry_path = artifacts_dir / "model_registry.json"
        save_model_registry(selected, registry_path, reason)
        
        # Print final console report
        print("\n" + "="*50)
        print("                 Model Comparison")
        print("="*50)
        print(f"{'Rank':<6}{'Model':<30}{'ROC-AUC':<10}{'F1':<8}{'Recall':<8}")
        print("-" * 62)
        for rank, model in enumerate(ranked_models, 1):
            print(f"{rank:<6}{model['model_name']:<30}{model['roc_auc']:.4f}    {model['f1_score']:.4f}  {model['recall']:.4f}")
            
        print("\nSelected Model:")
        print(selected["model_name"])
        
        print("\nReason:")
        print(reason)
        
        print("\nRegistry saved to:")
        print(registry_path)
        print("Comparison CSV saved to:")
        print(csv_path)
        print("="*50)
        
        logger.info("Model comparison and registry pipeline execution finished.")
        
    except Exception as e:
        logger.exception("An error occurred during model selection:")
        sys.exit(1)

if __name__ == "__main__":
    main()
