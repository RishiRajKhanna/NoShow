
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import os

# --- 1. SETUP & MODEL LOADING ---

# Define paths relative to the current script
# This makes it work even when run from a different directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'output') # Go up one level to the 'ds' folder, then into 'output'

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

# --- 2. DATA MODELS ---

# This defines the structure of the request the API expects
class Appointment(BaseModel):
    appointment_date: str # e.g., "2025-12-25"
    appointment_time: str # e.g., "14:30"
    past_noshow_rate: float
    days_since_last_appt: int
    duration_min: int
    appointment_type: str # e.g., "Examination"
    doctor: str # e.g., "Dr. Smith"

# --- 3. API CREATION ---

app = FastAPI()

# Add CORS middleware to allow requests from our frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],   # Allows all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the No-Show Prediction API"}

@app.post("/predict/")
def predict_no_show(appointment: Appointment):
    """Receives appointment data, preprocesses it, and returns a no-show prediction."""

    # --- 4. FEATURE ENGINEERING ---

    # Combine date and time and calculate lead time
    appt_datetime_str = f"{appointment.appointment_date} {appointment.appointment_time}"
    appt_datetime = datetime.strptime(appt_datetime_str, "%Y-%m-%d %H:%M")
    lead_time = appt_datetime - datetime.now()
    lead_time_hours = lead_time.total_seconds() / 3600

    # Hardcoded assumptions for simplicity
    # In a real app, this would come from a database
    resource_rates = {"Dr. Smith": 0.15, "Dr. Jones": 0.18, "Dr. Miller": 0.12}
    practice_noshow_rate = 0.1 # Clinic-wide average

    # Create the record dictionary for the model
    record = {
        'LEAD_TIME_HOURS': lead_time_hours,
        'DURATION_MIN': appointment.duration_min,
        'DAYS_SINCE_LAST_APPT': appointment.days_since_last_appt,
        'PAST_NOSHOW_RATE': appointment.past_noshow_rate / 100.0, # Convert from % to decimal
        'RESOURCE_NOSHOW_RATE': resource_rates.get(appointment.doctor, practice_noshow_rate),
        'PRACTICE_NOSHOW_RATE': practice_noshow_rate,
        'DAY_OF_WEEK': appt_datetime.strftime('%A'),
        'HOUR_OF_DAY': appt_datetime.hour,
        'APPOINTMENT_TYPE': appointment.appointment_type,
        'IS_WEEKEND': appt_datetime.weekday() >= 5
    }

    # --- 5. PREPROCESSING & PREDICTION ---

    df = pd.DataFrame([record])

    # Convert boolean to string for consistent encoding
    df['IS_WEEKEND'] = df['IS_WEEKEND'].astype(str)

    # One-Hot Encode
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Scale numerical features
    features_to_scale = [f for f in numerical_features if f in df.columns]
    if features_to_scale:
        df[features_to_scale] = scaler.transform(df[features_to_scale])

    # Make prediction
    probability = model.predict_proba(df)[0]
    no_show_prob = probability[1]

    return {
        "prediction": "No-Show" if no_show_prob > 0.5 else "Show",
        "no_show_probability": round(no_show_prob, 2)
    }
