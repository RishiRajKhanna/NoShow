from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np

# --- 1. SETUP & MODEL LOADING ---

# Define paths relative to the current script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
DATA_FILE = os.path.join(MODEL_DIR, 'final_features_and_eda.csv')

MODEL_FILE = os.path.join(MODEL_DIR, 'noshow_model_rf.joblib')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.joblib')
COLUMNS_FILE = os.path.join(MODEL_DIR, 'model_columns.joblib')

# Load artifacts
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
model_columns = joblib.load(COLUMNS_FILE)

# These are the original numerical features the scaler was trained on
numerical_features = [
    'LEAD_TIME_HOURS',
    'DURATION_MIN',
    'DAYS_SINCE_LAST_APPT',
    'PAST_NOSHOW_RATE',
    'RESOURCE_NOSHOW_RATE',
    'PRACTICE_NOSHOW_RATE'
]

# --- 2. API CREATION ---

app = FastAPI()

# Add CORS middleware to allow requests from our frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],   # Allows all headers
)

# --- 3. IMAGINary NAMES ---
# Create a mapping for patient IDs to imaginary names
patient_names = {
    "3134695-943": "John Doe",
    "47567-52227": "Jane Smith",
    "17239-1359": "Peter Jones",
    "22982-1359": "Mary Williams",
    "23399-1359": "David Brown",
    "12286-1359": "Susan Miller",
    "9718-1359": "Robert Davis",
    "1684-93989": "Patricia Garcia",
    "default": "Patient"
}

# --- 4. API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the No-Show Prediction API"}

@app.get("/get_appointments_by_date/")
def get_appointments_by_date(date: str):
    """
    Receives a date, loads the feature-engineered data, and returns a list of
    appointments for that day with no-show predictions and risk factors.
    """
    # --- Load Data ---
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Data file not found at {DATA_FILE}. Please run the data processing scripts first.")

    # --- Filter by Date and Sort ---
    df['APPOINTMENT_DATETIME'] = pd.to_datetime(df['APPOINTMENT_DATETIME'])
    df['APPOINTMENT_DATE'] = df['APPOINTMENT_DATETIME'].dt.date
    selected_date = datetime.strptime(date, "%Y-%m-%d").date()
    day_appointments = df[df['APPOINTMENT_DATE'] == selected_date].copy()
    day_appointments = day_appointments.sort_values(by='APPOINTMENT_DATETIME').copy()

    if day_appointments.empty:
        return {
            "summary": {
                "total_appointments": 0,
                "predicted_noshows": 0,
                "noshow_rate": 0,
                "top_risk_factors": "N/A"
            },
            "appointments": []
        }

    # --- Data Preparation for Model ---
    # Convert boolean 'IS_WEEKEND' to object type for consistent encoding
    day_appointments['IS_WEEKEND'] = day_appointments['IS_WEEKEND'].astype(str)

    # One-Hot Encode categorical features
    X_day = pd.get_dummies(day_appointments, columns=['DAY_OF_WEEK', 'HOUR_OF_DAY', 'APPOINTMENT_TYPE', 'IS_WEEKEND'])

    # Align columns with the model's training columns
    X_day = X_day.reindex(columns=model_columns, fill_value=0)

    # --- Feature Scaling ---
    X_day_scaled = X_day.copy()
    X_day_scaled[numerical_features] = scaler.transform(X_day[numerical_features])

    # --- Prediction and Risk Analysis ---
    appointments_list = []
    high_risk_count = 0
    all_risk_factors = []

    # Get predictions for all appointments for the day
    probabilities = model.predict_proba(X_day_scaled)
    no_show_probabilities = probabilities[:, 1]

    day_appointments['NO_SHOW_PROBABILITY'] = no_show_probabilities

    for index, row in day_appointments.iterrows():
        probability = row['NO_SHOW_PROBABILITY']
        risk_factors = []

        # Categorize Risk
        if probability > 0.6:
            risk_level = "High Risk"
            high_risk_count += 1
        elif 0.3 <= probability <= 0.6:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        # Identify Risk Factors
        if row['PAST_NOSHOW_RATE'] > 0.5:
            risk_factors.append("History of No-Shows")
        if row['LEAD_TIME_HOURS'] > 72:
            risk_factors.append("Booked Far in Advance")
        if row['APPOINTMENT_TYPE'] in ["Boarding", "Grooming"]:
            risk_factors.append("High-Risk Appointment Type")
        if risk_level == "High Risk":
            all_risk_factors.extend(risk_factors)

        appointments_list.append({
            "id": row['APPOINTMENT_ODU_ID'],
            "patient_name": patient_names.get(row['PATIENT_ODU_ID'], patient_names["default"]),
            "time": row['APPOINTMENT_DATETIME'].strftime("%I:%M %p"),
            "reason": row['APPOINTMENT_TYPE'],
            "risk_factors": ", ".join(risk_factors) if risk_factors else "None",
            "prediction": risk_level,
            "action": "📞" if risk_level == "High Risk" else ("✉️" if risk_level == "Medium Risk" else "✅")
        })

    # --- Calculate Summary ---
    total_appointments = len(day_appointments)
    noshow_rate = (high_risk_count / total_appointments) * 100 if total_appointments > 0 else 0

    if all_risk_factors:
        top_risk_factors = pd.Series(all_risk_factors).value_counts().index[0]
    else:
        top_risk_factors = "None"

    summary = {
        "total_appointments": total_appointments,
        "predicted_noshows": high_risk_count,
        "noshow_rate": round(noshow_rate, 1),
        "top_risk_factors": top_risk_factors
    }

    return {
        "summary": summary,
        "appointments": appointments_list
    }