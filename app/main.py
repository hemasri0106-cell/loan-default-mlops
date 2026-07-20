import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Ensure project root and app directory are in sys.path
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
BASE_DIR = APP_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

try:
    from app.predictor import get_predictor_service
except ImportError:
    from predictor import get_predictor_service

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction System",
    description="AI-powered loan default prediction using machine learning and automated model monitoring.",
    version="2.0.0"
)

# Mount static files and Jinja2 templates
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Input Schema Validation
class LoanPredictionInput(BaseModel):
    # Borrower Info
    annual_inc: float = Field(..., example=75000.0, description="Annual Income ($)")
    emp_length: str = Field(..., example="10+ years", description="Employment Length")
    home_ownership: str = Field(..., example="MORTGAGE", description="Home Ownership")
    addr_state: str = Field(..., example="CA", description="State Code")
    
    # Loan Details
    loan_amnt: float = Field(..., example=15000.0, description="Loan Amount ($)")
    term: str = Field(..., example=" 36 months", description="Loan Term")
    int_rate: float = Field(..., example=11.5, description="Interest Rate (%)")
    installment: float = Field(..., example=350.0, description="Monthly Installment ($)")
    purpose: str = Field(..., example="debt_consolidation", description="Loan Purpose")
    issue_d: str = Field(..., example="2018-01-01", description="Issue Date (YYYY-MM-DD)")
    
    # Credit History
    fico_range_low: float = Field(..., example=700.0, description="FICO Range Low")
    fico_range_high: float = Field(..., example=704.0, description="FICO Range High")
    open_acc: float = Field(..., example=11.0, description="Open Credit Lines")
    revol_util: float = Field(..., example=45.0, description="Revolving Line Utilization (%)")
    
    # Ratios & Ratings
    dti: float = Field(..., example=16.5, description="Debt-To-Income Ratio (%)")
    grade: str = Field(..., example="B", description="Credit Grade")
    sub_grade: str = Field(..., example="B2", description="Credit Sub-Grade")
    verification_status: str = Field(..., example="Source Verified", description="Income Verification Status")

@app.on_event("startup")
async def startup_event():
    """Warm up and load predictor service artifacts at application startup."""
    logger.info("Initializing FastAPI application...")
    try:
        service = get_predictor_service()
        info = service.get_model_info()
        logger.info(f"Deployed Champion Model Active: {info['selected_model']} (Version {info['version']})")
    except Exception as e:
        logger.error(f"Error initializing predictor service on startup: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main FinTech Credit Risk Prediction Dashboard."""
    try:
        service = get_predictor_service()
        model_info = service.get_model_info()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"model_info": model_info}
        )
    except Exception as e:
        logger.error(f"Error rendering homepage: {e}")
        return HTMLResponse(content=f"<h2>System Initialization Error</h2><p>{str(e)}</p>", status_code=500)

@app.post("/predict")
async def predict_loan_risk(payload: LoanPredictionInput):
    """API endpoint for loan default prediction."""
    try:
        service = get_predictor_service()
        input_dict = payload.dict()
        
        logger.info(f"Prediction request received for loan_amnt=${payload.loan_amnt:,.2f}, annual_inc=${payload.annual_inc:,.2f}")
        result = service.predict(input_dict)
        
        logger.info(f"Prediction result: '{result['prediction']}' | Risk Score: {result['risk_score']}% | Latency: {result['latency_ms']}ms")
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        logger.exception("Error processing prediction request:")
        return JSONResponse(
            content={
                "error": True,
                "message": "An error occurred while computing default risk. Please verify feature inputs and try again.",
                "details": str(e)
            },
            status_code=400
        )

@app.get("/api/model-info")
async def get_model_info_api():
    """API endpoint returning deployed champion model metadata from model registry."""
    try:
        service = get_predictor_service()
        return JSONResponse(content=service.get_model_info(), status_code=200)
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  Starting Credit Risk Web Server at http://127.0.0.1:8000")
    print("="*50 + "\n")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
