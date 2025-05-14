from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import uvicorn
from functools import lru_cache

# Create an instance of FastAPI
app = FastAPI(
    title="Loan Approval API",
    description="API for predicting loan approvals using XGBoost model",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache the model loading for reuse
@lru_cache(maxsize=1)
def get_model():
    model = xgb.XGBClassifier()
    model.load_model('xgboost_model.json')
    return model

# Pre-define encoders with their classes
home_ownership_encoder = LabelEncoder()
home_ownership_encoder.fit(['rent', 'own', 'mortgage', 'other'])  # Fit the encoder

previous_defaults_encoder = LabelEncoder()
previous_defaults_encoder.fit(['yes', 'no'])  # Fit the encoder

# Cache valid values for fast validation
HOME_OWNERSHIP_VALID_VALUES = set(home_ownership_encoder.classes_)
DEFAULTS_VALID_VALUES = set(previous_defaults_encoder.classes_)

# Define column mapping for consistent naming
COLUMN_MAPPING = {
    'person_age': 'age',
    'person_income': 'income',
    'loan_amnt': 'loan_amount',
    'loan_int_rate': 'loan_int_rate',
    'previous_loan_defaults_on_file': 'previous_defaults',
    'person_home_ownership': 'home_ownership'
}

# Input validation models
class LoanApplication(BaseModel):
    income: float = Field(..., gt=0, description="Annual income")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(..., ge=5, le=20, description="Loan Interest Rate (5-20%)")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    previous_defaults: str = Field(..., description="Previous loan defaults on file (yes/no)")
    home_ownership: str = Field(..., description="Home ownership status (rent/own/mortgage/other)")
    
    @validator('previous_defaults')
    def validate_previous_defaults(cls, v):
        v_lower = v.lower().strip()
        if v_lower not in DEFAULTS_VALID_VALUES:
            raise ValueError(f"Invalid previous defaults value. Must be one of: {', '.join(DEFAULTS_VALID_VALUES)}")
        return v_lower
    
    @validator('home_ownership')
    def validate_home_ownership(cls, v):
        v_lower = v.lower().strip()
        if v_lower not in HOME_OWNERSHIP_VALID_VALUES:
            raise ValueError(f"Invalid home ownership value. Must be one of: {', '.join(HOME_OWNERSHIP_VALID_VALUES)}")
        return v_lower

@app.get("/")
def read_root():
    return {
        "message": "Loan Approval API is running",
        "endpoints": {
            "/predict/csv": "Upload CSV file for batch predictions",
            "/analyze/csv": "Upload CSV file for analysis and visualization data"
        }
    }

def process_loan_data(df: pd.DataFrame, model):
    """Process loan data and make predictions - extracted common functionality"""
    # Standardize column names (ensure all columns are properly mapped)
    df = df.rename(columns={old: new for old, new in COLUMN_MAPPING.items() if old in df.columns})
    
    # Ensure that all necessary columns are present
    required_columns = ['previous_defaults', 'loan_amount', 'loan_int_rate', 'age', 'income', 'home_ownership']
    
    # Check if all required columns are present
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_cols)}")

    # Clean categorical columns (vectorized operation)
    df['home_ownership'] = df['home_ownership'].str.strip().str.lower()
    df['previous_defaults'] = df['previous_defaults'].str.strip().str.lower()
    
    # Quick validation
    invalid_ownership = ~df['home_ownership'].isin(HOME_OWNERSHIP_VALID_VALUES)
    if invalid_ownership.any():
        invalid_values = df.loc[invalid_ownership, 'home_ownership'].unique().tolist()
        raise HTTPException(status_code=400, 
                           detail=f"Invalid home ownership types: {invalid_values}")
    
    invalid_defaults = ~df['previous_defaults'].isin(DEFAULTS_VALID_VALUES)
    if invalid_defaults.any():
        invalid_values = df.loc[invalid_defaults, 'previous_defaults'].unique().tolist()
        raise HTTPException(status_code=400, 
                           detail=f"Invalid previous loan default types: {invalid_values}")
    
    # Transform categorical features to numeric
    df['home_ownership_encoded'] = home_ownership_encoder.transform(df['home_ownership'])
    df['previous_defaults_encoded'] = previous_defaults_encoder.transform(df['previous_defaults'])
    
    # Calculate engineered features efficiently
    df['debt_to_income_ratio'] = df['loan_amount'] / df['income']
    df['age_income_ratio'] = df['age'] / df['income']
    
    # Prepare model input features
    # Ensure all 6 columns are passed to the model
    features = df[[
        'previous_defaults_encoded',
        'debt_to_income_ratio',
        'loan_int_rate',
        'age_income_ratio',
        'home_ownership_encoded',
        'loan_amount'  # Add this feature to match model's expected input shape
    ]].values.astype(np.float32)

    # Predict
    probabilities = model.predict_proba(features)
    df['result'] = np.where(probabilities[:, 1] > 0.5, 'Approved', 'Declined')
    df['approval_probability'] = np.round(probabilities[:, 1] * 100, 2)
    
    return df


@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(50, description="Items per page", ge=10, le=1000),
    chunk_size: int = Query(1000, description="Number of rows to process at once"),
    model=Depends(get_model)
):
    """Process CSV in chunks for memory efficiency with large files and return paginated results"""
    # Validate file extension
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Process CSV in chunks
        results = []
        content = await file.read()
        csv_file = io.StringIO(content.decode('utf-8'))
        
        # Read the CSV in chunks
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            # Process the chunk
            processed_chunk = process_loan_data(chunk, model)
            
            # Add to results
            chunk_result = processed_chunk[[
                'age', 'income', 'loan_amount', 'loan_int_rate', 'loan_percent_income',
                'previous_defaults', 'home_ownership',
                'result', 'approval_probability'
            ]]
            
            results.append(chunk_result)
        
        # Combine all chunks
        if not results:
            return {
                "results": [],
                "pagination": {
                    "page": 1,
                    "page_size": page_size,
                    "total_items": 0,
                    "total_pages": 0
                }
            }

        combined_results = pd.concat(results, ignore_index=True)

        total_items = len(combined_results)
        total_pages = (total_items + page_size - 1) // page_size

        # Clamp page value
        page = min(page, total_pages) if total_pages > 0 else 1

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # Use direct slicing instead of iloc when index is simple range
        page_data = combined_results[start_idx:end_idx]

        return {
            "results": page_data.to_dict(orient="records"),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": total_pages
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/analyze/csv")
async def analyze_csv(
    file: UploadFile = File(...),
    model=Depends(get_model)
):
    """Analyze CSV data and return summary statistics for visualization"""
    # Validate file extension
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        csv_file = io.StringIO(content.decode('utf-8'))
        df = pd.read_csv(csv_file)
        
        # Process data
        df = process_loan_data(df, model)
        
        # Get overall approval rate
        approval_rate = (df['result'] == 'Approved').mean() * 100
        
        # Create summary statistics for visualization
        summary = {
            "total_applications": len(df),
            "approval_rate": round(approval_rate, 2),
            
            # Approval rate by home ownership
            "approval_by_ownership": pd.crosstab(
                df['home_ownership'], 
                df['result'],
                normalize='index'
            ).round(4).to_dict(orient="index"),
            
            # Approval rate by previous loan defaults
            "approval_by_defaults": pd.crosstab(
                df['previous_defaults'],
                df['result'],
                normalize='index'
            ).round(4).to_dict(orient="index")
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="0.0.0.0", port=8080)
