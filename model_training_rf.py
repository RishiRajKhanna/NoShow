
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Define paths
INPUT_FILE = os.path.join('output', 'final_features_and_eda.csv')
OUTPUT_DIR = 'output'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'noshow_model_rf.joblib') # New model file name
COLUMNS_FILE = os.path.join(OUTPUT_DIR, 'model_columns.joblib')
SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler.joblib')

# Load the dataset
print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# --- Feature Engineering & Selection ---
print("Preparing data for modeling...")

numerical_features = [
    'LEAD_TIME_HOURS',
    'DURATION_MIN',
    'DAYS_SINCE_LAST_APPT',
    'PAST_NOSHOW_RATE',
    'RESOURCE_NOSHOW_RATE',
    'PRACTICE_NOSHOW_RATE'
]

categorical_features = [
    'DAY_OF_WEEK',
    'HOUR_OF_DAY',
    'APPOINTMENT_TYPE',
    'IS_WEEKEND'
]

target = 'NO_SHOW'

features = numerical_features + categorical_features
df_model = df[features + [target]].copy()

# --- Data Preprocessing ---

# Handle missing values
for col in numerical_features:
    df_model[col].fillna(0, inplace=True)

# Convert boolean 'IS_WEEKEND' to object type for consistent encoding
df_model['IS_WEEKEND'] = df_model['IS_WEEKEND'].astype(str)

# Convert target variable to numeric (0 or 1)
le = LabelEncoder()
df_model[target] = le.fit_transform(df_model[target])

# One-Hot Encode categorical features
df_model = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

# Separate features (X) and target (y)
X = df_model.drop(target, axis=1)
y = df_model[target]

# Load the saved columns to ensure consistency
model_columns = joblib.load(COLUMNS_FILE)
X = X[model_columns]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Feature Scaling ---
print("Scaling numerical features...")
scaler = StandardScaler()

# Create copies to avoid SettingWithCopyWarning
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Fit and transform the numerical features
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])


# --- Model Training (Random Forest) ---
print("Training Random Forest model on scaled data...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train_scaled, y_train)

# --- Model Evaluation ---
print("Evaluating model performance on scaled data...")
y_pred = model.predict(X_test_scaled)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Show', 'No-Show'])

print(f"\nModel Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)

# --- Save the Model & Scaler ---
joblib.dump(model, MODEL_FILE)
# We can reuse the same scaler, but saving it again with the new model is fine.
joblib.dump(scaler, SCALER_FILE) 
print(f"\nModel saved to {MODEL_FILE}")
print(f"Scaler saved to {SCALER_FILE}")
print("Random Forest model training and evaluation complete.")
