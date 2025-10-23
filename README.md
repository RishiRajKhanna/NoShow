# Appointment No-Show Prediction Project

## 1. Objective

The primary objective of this project was to analyze historical appointment and transaction data to build a machine learning model capable of predicting whether a patient will "no-show" for a scheduled appointment. The goal is to provide actionable insights that can help reduce revenue loss from missed appointments.

---

## 2. Project Structure

The project is organized into a series of scripts that perform sequential tasks. All generated outputs, including datasets, plots, and trained models, are stored in the `output/` directory.

- `Apointment_clean.csv`, `Transaction_clean.csv`: Raw input data.
- `analysis.py`: Script for initial data merging and creation of the `NO_SHOW` label.
- `feature_engineering_eda.py`: Script for creating new features and performing exploratory data analysis (EDA).
- `model_training.py`: Script for training and evaluating a baseline **Logistic Regression** model.
- `model_training_rf.py`: Script for training and evaluating the final **Random Forest** model.
- `output/`: Directory for all generated files.
- `README.md`: This documentation file.

---

## 3. Methodology

The project was executed in three main phases:

### Phase 1: Core Data Processing (`analysis.py`)

This phase focused on merging the two source files and creating the ground truth `NO_SHOW` label.

- **Logic**: An appointment was labeled as a `NO_SHOW` (`True`) unless there was clear evidence the patient attended. This evidence required three conditions to be met:
    1. The patient had an in-clinic transaction (`ANY_INCLINIC` = `True`).
    2. The patient's transaction generated revenue (`ANY_REVENUE` = `True`).
    3. The appointment was not deleted (`ODU_IS_DELETED` = `False`).
- **Output**: A clean, intermediate dataset named `analysis_results.csv`.

### Phase 2: Feature Engineering & EDA (`feature_engineering_eda.py`)

This phase focused on enriching the dataset with powerful features for prediction and generating business insights.

- **Key Features Created**:
    - `LEAD_TIME_HOURS`: Time between appointment creation and the scheduled time.
    - `DAYS_SINCE_LAST_APPT`: Time since this patient's previous appointment.
    - `PAST_NOSHOW_RATE`: The historical no-show rate for the specific patient.
    - `RESOURCE_NOSHOW_RATE`: The historical no-show rate for the specific doctor/resource.
    - `PRACTICE_NOSHOW_RATE`: The historical no-show rate for the entire clinic.
    - Time-based features like `DAY_OF_WEEK`, `HOUR_OF_DAY`, etc.
- **Outputs**:
    - `final_features_and_eda.csv`: The final, fully-enriched dataset ready for modeling.
    - A series of plots saved to the `output/` directory, such as `noshow_by_day.png` and `noshow_heatmap.png`.

### Phase 3: Predictive Modeling

In this phase, we selected the best features, prepared the data, and trained two different models to compare their performance.

- **Feature Selection for Modeling**:
    - **Included Features**: `LEAD_TIME_HOURS`, `DURATION_MIN`, `DAYS_SINCE_LAST_APPT`, `PAST_NOSHOW_RATE`, `RESOURCE_NOSHOW_RATE`, `PRACTICE_NOSHOW_RATE`, `DAY_OF_WEEK`, `HOUR_OF_DAY`, `APPOINTMENT_TYPE`, `IS_WEEKEND`.
    - **Excluded Features**: Fields like `ANY_INCLINIC` and `TOTAL_AMOUNT` were excluded from the model features because they are only known *after* an appointment occurs and would cause data leakage.

- **Data Preparation Steps**:
    1. **Handled Missing Values**: Filled `NaN`s with `0` (e.g., for a patient's first appointment).
    2. **Encoded Categorical Features**: Converted text-based features like `DAY_OF_WEEK` into a numerical format using one-hot encoding.
    3. **Scaled Numerical Features**: Used `StandardScaler` to put all numerical features on a common scale, improving model stability.

--- 

## 4. Model Comparison & Results

Two models were trained and evaluated to find the best performer.

### Experiment 1: Logistic Regression (Baseline)

- **Why**: Chosen as a simple, interpretable baseline model.
- **Script**: `model_training.py`
- **Performance**:
    - **Accuracy**: 75.8%
    - **No-Show Precision**: 52% (Correctness of a "no-show" prediction)
    - **No-Show Recall**: 75% (Ability to find all actual no-shows)

### Experiment 2: Random Forest (Final Model)

- **Why**: Chosen to capture more complex patterns in the data and improve upon the baseline, with a goal of increasing precision.
- **Script**: `model_training_rf.py`
- **Performance**:
    - **Accuracy**: 82.2%
    - **No-Show Precision**: 71%
    - **No-Show Recall**: 51%

### Conclusion

| Metric                 | Logistic Regression | **Random Forest (Winner)** |
| :--------------------- | :------------------ | :------------------------- |
| **Overall Accuracy**   | 75.8%               | **82.2%**                  |
| **"No-Show" Precision**| 52%                 | **71%**                    |
| **"No-Show" Recall**   | 75%                 | 51%                        |

The **Random Forest model is the recommended model**. While it catches a smaller percentage of the total no-shows (lower recall), it is **far more precise** in its predictions. A 71% precision rate means that when the model flags a patient as a high risk, that prediction is correct nearly 3 out of 4 times, making it a much more reliable and actionable tool for business interventions.

--- 

## 5. How to Run the Project

To replicate the analysis, run the scripts from the command line in the following order:

```bash
# 1. Initial data processing
python analysis.py

# 2. Feature engineering and EDA
python feature_engineering_eda.py

# 3. Train and evaluate the baseline Logistic Regression model
python model_training.py

# 4. Train and evaluate the final Random Forest model
python model_training_rf.py
```

### 6. Prediction Tool (Frontend & Backend)

To make the no-show prediction model accessible, a web-based tool has been developed, consisting of a FastAPI backend and a simple HTML/CSS/JavaScript frontend.

#### 6.1 Backend API (`api/main.py`)

The backend is a FastAPI application that serves as the prediction engine.

-   **Purpose**: Receives appointment details from the frontend, preprocesses the data, uses the trained Random Forest model to make a no-show prediction, and returns the result.
-   **Key Logic**:
    -   Loads the `noshow_model_rf.joblib` (Random Forest model), `scaler.joblib` (feature scaler), and `model_columns.joblib` (list of expected features) from the `output/` directory.
    -   Exposes a `/predict` endpoint that accepts `POST` requests with appointment data.
    -   Performs feature engineering (e.g., calculates `LEAD_TIME_HOURS`, `DAY_OF_WEEK`).
    -   Applies one-hot encoding and scaling to the input data, ensuring consistency with the model's training.
    -   Returns a prediction (`"No-Show"` or `"Show"`) and the probability of a no-show.
-   **Dependencies**: Listed in `api/requirements.txt`.
-   **How to Run**:
    1.  Navigate to the `api/` directory:
        ```bash
        cd api
        ```
    2.  Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    3.  Start the FastAPI server:
        ```bash
        python -m uvicorn main:app --host 0.0.0.0 --port 8000
        ```
        Leave this terminal running.

#### 6.2 Frontend Web Interface (`frontend/`)

The frontend provides a user-friendly interface for interacting with the prediction API.

-   **Purpose**: Allows a receptionist to input appointment details and instantly receive a no-show risk prediction.
-   **Files**:
    -   `index.html`: The main HTML structure for the prediction form.
    -   `script.js`: Handles form submission, sends data to the backend API, and displays the prediction results.
    -   `style.css`: Provides styling for the web interface.
-   **How to Run**:
    1.  Ensure the backend API is running (as described above).
    2.  Navigate to the `frontend/` directory in your file explorer.
    3.  Open the `index.html` file directly in your web browser (e.g., by double-clicking it).

The web interface will connect to the running backend API to fetch predictions.