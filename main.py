from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load the model
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')

# Define the label encoder for the categorical variables
label_encoder = LabelEncoder()
label_encoder.fit(['rent', 'own', 'mortgage'])  # Fit the encoder to these categories

@app.get("/")
def read_root():
    return {"Hello": "World"}

class LoanApplication(BaseModel):
    income: float
    loan_amount: float
    credit_score: float
    age: int
    previous_loan_defaults_on_file: float
    person_home_ownership: str  # Categorical feature: rent, own, mortgage

@app.post("/predict")
def predict(data: LoanApplication):
    # Normalize and clean the person_home_ownership string (strip whitespace and lowercase)
    home_ownership = data.person_home_ownership.strip().lower()
    
    # Check if the input string is valid
    if home_ownership not in label_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Invalid home ownership type: '{home_ownership}'. Valid types are 'rent', 'own', 'mortgage'.")
    
    # Label encode the person_home_ownership
    person_home_ownership_encoded = label_encoder.transform([home_ownership])[0]

    # Calculate the necessary ratios
    debt_to_income_ratio = data.loan_amount / data.income if data.income > 0 else 0
    loan_percent_income = (data.loan_amount / data.income) * 100  # Percentage
    age_income_ratio = data.age / data.income if data.income > 0 else 0
    
    # Construct the input array with calculated ratios and encoded feature
    input_data = np.array([[
        data.previous_loan_defaults_on_file, 
        debt_to_income_ratio, 
        loan_percent_income, 
        data.credit_score,  # Assuming credit score directly used
        age_income_ratio, 
        person_home_ownership_encoded  # Encoded categorical feature
    ]])

    # Get the probabilities for each class (0 = Declined, 1 = Approved)
    proba = model.predict_proba(input_data)[0]

    # Get the probability of approval (class 1) and convert it to percentage
    approval_probability = float(proba[1]) * 100  # Explicitly convert to float

    # Make the prediction (class with the higher probability)
    prediction = model.predict(input_data)

    # Based on the prediction, return the result
    result = "Approved" if prediction[0] == 1 else "Declined"
    
    return {
        "result": result,
        "approval_probability": round(approval_probability, 2)  # Showing the percentage with two decimals
    }
