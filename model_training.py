
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Define paths
INPUT_FILE = os.path.join('output', 'final_features_and_eda.csv')
OUTPUT_DIR = 'output'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'noshow_model.joblib')
COLUMNS_FILE = os.path.join(OUTPUT_DIR, 'model_columns.joblib')

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the dataset
print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# --- Feature Engineering & Selection ---
print("Preparing data for modeling...")

# Select features
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
# For rate features, NaN likely means no prior data, so a rate of 0 is a reasonable default.
# For 'DAYS_SINCE_LAST_APPT', NaN means a first appointment, so 0 is also appropriate.
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

# Save the columns used for training
model_columns = X.columns.tolist()
joblib.dump(model_columns, COLUMNS_FILE)
print(f"Model columns saved to {COLUMNS_FILE}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Feature Scaling ---
# We scale the numerical features to ensure they have a similar impact on the model.
# The scaler is fit ONLY on the training data to prevent data leakage.
print("Scaling numerical features...")
scaler = StandardScaler()
SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler.joblib')

# Create copies to avoid SettingWithCopyWarning
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Fit and transform the numerical features
# Note: We only use the original 'numerical_features' list for this
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])


# --- Model Training ---
print("Training Logistic Regression model on scaled data...")
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
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
joblib.dump(scaler, SCALER_FILE)
print(f"\nModel saved to {MODEL_FILE}")
print(f"Scaler saved to {SCALER_FILE}")
print("Model training and evaluation complete.")
