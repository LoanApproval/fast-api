from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import io
import csv
import uvicorn
import os
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=os.getenv("PORT", default=8080), log_level="info")

app = FastAPI()

# Load the model
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')

# Define the label encoder for the categorical variables
label_encoder = LabelEncoder()
label_encoder.fit(['rent', 'own', 'mortgage','other'])  # Fit the encoder to these categories

previous_loan_defaults_encoder = LabelEncoder()
previous_loan_defaults_encoder.fit(['yes', 'no'])  # Fit the encoder to these categories

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
    
@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    csv_file = io.StringIO(contents.decode('utf-8'))
    df = pd.read_csv(csv_file)

    # Lowercase และ strip ค่า categorical ล่วงหน้า
    df['person_home_ownership'] = df['person_home_ownership'].str.strip().str.lower()
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].str.strip().str.lower()

    # ตรวจสอบค่า invalid
    invalid_home_ownership = ~df['person_home_ownership'].isin(label_encoder.classes_)
    if invalid_home_ownership.any():
        raise HTTPException(status_code=400, detail=f"Invalid home ownership types: {df['person_home_ownership'][invalid_home_ownership].unique().tolist()}")

    invalid_defaults = ~df['previous_loan_defaults_on_file'].isin(previous_loan_defaults_encoder.classes_)
    if invalid_defaults.any():
        raise HTTPException(status_code=400, detail=f"Invalid previous loan default types: {df['previous_loan_defaults_on_file'][invalid_defaults].unique().tolist()}")

    # แปลงเป็นตัวเลขด้วย LabelEncoder
    df['person_home_ownership_encoded'] = label_encoder.transform(df['person_home_ownership'])
    df['previous_loan_defaults_encoded'] = previous_loan_defaults_encoder.transform(df['previous_loan_defaults_on_file'])

    # คำนวณฟีเจอร์ใหม่แบบเวกเตอร์
    df['debt_to_income_ratio'] = df['loan_amnt'] / df['person_income']
    df['loan_percent_income'] = (df['loan_amnt'] / df['person_income']) * 100
    df['age_income_ratio'] = df['person_age'] / df['person_income']

    # เตรียมข้อมูล input สำหรับโมเดล
    input_data = df[[
        'previous_loan_defaults_encoded',
        'debt_to_income_ratio',
        'loan_percent_income',
        'credit_score',
        'age_income_ratio',
        'person_home_ownership_encoded'
    ]].astype(np.float32).values

    # ทำนายผลลัพธ์
    proba = model.predict_proba(input_data)
    preds = model.predict(input_data)

    # สร้างผลลัพธ์
    df['result'] = np.where(preds == 1, 'Approved', 'Declined')
    df['approval_probability'] = np.round(proba[:, 1] * 100, 2)

    # เลือกคอลัมน์ที่จะส่งกลับ
    return {
        "results": df[[
            'person_age', 'person_income', 'loan_amnt', 'credit_score',
            'previous_loan_defaults_on_file', 'person_home_ownership',
            'result', 'approval_probability'
        ]].rename(columns={
            'person_age': 'age',
            'person_income': 'income',
            'loan_amnt': 'loan_amount'
        }).to_dict(orient="records")
    }
