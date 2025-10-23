
import pandas as pd
import joblib
import os

# --- 1. LOAD SAVED ARTIFACTS ---

OUTPUT_DIR = 'output'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'noshow_model_rf.joblib')
SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler.joblib')
COLUMNS_FILE = os.path.join(OUTPUT_DIR, 'model_columns.joblib')

print("Loading model and other artifacts...")
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

def predict_single(record):
    """
    Predicts the no-show probability for a single appointment record.
    Performs all the necessary preprocessing steps.
    """
    print("\nProcessing new appointment record...")
    # Convert single record dictionary to a DataFrame
    df = pd.DataFrame([record])

    # --- 2. PREPROCESS THE NEW RECORD (must match training steps) ---

    # Handle missing values
    for col in numerical_features:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    # Ensure boolean is string for consistent encoding
    if 'IS_WEEKEND' in df.columns:
        df['IS_WEEKEND'] = df['IS_WEEKEND'].astype(str)

    # One-Hot Encode categorical features
    df = pd.get_dummies(df)

    # Align columns with the model's training columns
    # This is a crucial step: adds missing columns (with value 0) and removes extra ones.
    df = df.reindex(columns=model_columns, fill_value=0)

    # Scale numerical features using the loaded scaler
    # We only scale the columns that are both in the record and in the numerical_features list
    features_to_scale = [f for f in numerical_features if f in df.columns]
    if features_to_scale:
        df[features_to_scale] = scaler.transform(df[features_to_scale])

    print("Preprocessing complete. Making prediction...")

    # --- 3. MAKE PREDICTION ---

    # Predict the probability [P(Show), P(No-Show)]
    probability = model.predict_proba(df)[0]
    prediction = model.predict(df)[0]

    return prediction, probability


# --- 4. EXAMPLE USAGE ---

if __name__ == "__main__":
    # Create a sample new appointment record.
    # You can change these values to test different scenarios.
    new_appointment = {
        'LEAD_TIME_HOURS': 300,         # Booked 300 hours in advance
        'DURATION_MIN': 15,             # A 15-minute appointment
        'DAYS_SINCE_LAST_APPT': 90,     # Patient's last visit was 90 days ago
        'PAST_NOSHOW_RATE': 0.5,        # Patient has a 50% no-show history
        'RESOURCE_NOSHOW_RATE': 0.15,   # Doctor has a 15% no-show rate
        'PRACTICE_NOSHOW_RATE': 0.1,    # Clinic has a 10% no-show rate
        'DAY_OF_WEEK': 'Monday',        # Appointment is on a Monday
        'HOUR_OF_DAY': 10,              # at 10 AM
        'APPOINTMENT_TYPE': 'Examination',
        'IS_WEEKEND': False
    }

    # Get the prediction
    pred_label, pred_prob = predict_single(new_appointment)

    # Interpret and print the result
    no_show_probability = pred_prob[1] # Probability of the 'No-Show' class

    print("\n--- PREDICTION RESULT ---")
    if pred_label == 1:
        print(f"Prediction: PATIENT WILL LIKELY NO-SHOW")
    else:
        print(f"Prediction: Patient will likely show up")

    print(f"Confidence (Probability of No-Show): {no_show_probability:.0%}")
    print("-------------------------")
