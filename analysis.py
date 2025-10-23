
import pandas as pd
import os

# Step 1 — Memory-saving preliminaries
print("Step 1: Reading data with memory optimization...")

appt_cols = [
 'APPOINTMENT_ODU_ID','PATIENT_ODU_ID','APPOINTMENT_DATE','APPOINTMENT_DATETIME',
 'DURATION','PIMS_SOURCE','PIMS_SCHEDULE_TYPE','PIMS_STATUS','NOTES',
 'APPOINTMENT_TYPE','IS_CANCELED_APPOINTMENT','ODU_IS_DELETED',
 'CREATED_DATE', 'RESOURCE_ODU_ID', 'PRACTICE_ODU_ID', 'PRACTICE_NAME'
]
txn_cols = [
 'TRANSACTION_ODU_ID','PATIENT_ODU_ID','REPORTING_DATE','REPORTING_DATETIME',
 'REPORTING_AMOUNT','QUANTITY','IS_INCLINIC','IS_REVENUE','PIMS_TRANSACTION_TYPE',
 'IS_ONLINE','IS_PAYMENT','TOP_REVENUE_CATEGORY_NAME'
]

dtype_appt = {'APPOINTMENT_ODU_ID': 'string', 'PATIENT_ODU_ID': 'string',
             'PIMS_SOURCE':'category','PIMS_SCHEDULE_TYPE':'category',
             'PIMS_STATUS':'category','APPOINTMENT_TYPE':'category',
             'ODU_IS_DELETED': 'boolean'} # Explicitly set ODU_IS_DELETED to boolean
dtype_txn = {'TRANSACTION_ODU_ID':'string','PATIENT_ODU_ID':'string',
            'IS_INCLINIC':'boolean','IS_REVENUE':'boolean',
            'IS_ONLINE':'boolean','IS_PAYMENT':'boolean',
            'TOP_REVENUE_CATEGORY_NAME':'category'}

try:
    appts = pd.read_csv('Apointment_clean.csv', usecols=appt_cols, dtype=dtype_appt,
                        parse_dates=['APPOINTMENT_DATE','APPOINTMENT_DATETIME'])
    txns = pd.read_csv('Transaction_clean.csv', usecols=txn_cols, dtype=dtype_txn,
                       parse_dates=['REPORTING_DATE','REPORTING_DATETIME'])
    print("CSVs loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading CSVs: {e}")
    print("Please ensure 'Apointment_clean.csv' and 'Transaction_clean.csv' are in the directory.")
    exit()

# Filter out rows with missing PATIENT_ODU_ID from appointments
print("\nFiltering out entries with missing PATIENT_ODU_ID...")
initial_rows = len(appts)
appts.dropna(subset=['PATIENT_ODU_ID'], inplace=True)
final_rows = len(appts)
print(f"{initial_rows - final_rows} rows with missing PATIENT_ODU_ID were removed.")


# Step 2 — Aggregate transactions
print("\nStep 2: Aggregating transactions...")

appts['APPT_DATE'] = pd.to_datetime(appts['APPOINTMENT_DATE']).dt.date
txns['TXN_DATE'] = pd.to_datetime(txns['REPORTING_DATE']).dt.date

agg_txn = (
    txns
    .groupby(['PATIENT_ODU_ID','TXN_DATE'], as_index=False)
    .agg(
        total_amount=('REPORTING_AMOUNT', 'sum'),
        total_qty=('QUANTITY','sum'),
        any_inclinic=('IS_INCLINIC','max'),
        any_revenue=('IS_REVENUE','max'),
        txn_count=('TRANSACTION_ODU_ID','nunique')
    )
)
agg_txn['any_inclinic'] = agg_txn['any_inclinic'].astype(bool)
agg_txn['any_revenue'] = agg_txn['any_revenue'].astype(bool)
print("Transactions aggregated.")
print("Aggregated transactions head:")
print(agg_txn.head())

# Step 3 — Create the no_show label
print("\nStep 3: Creating the no_show label...")

agg_txn.rename(columns={'TXN_DATE':'APPT_DATE'}, inplace=True)

merged = appts.merge(agg_txn, how='left',
                     left_on=['PATIENT_ODU_ID','APPT_DATE'],
                     right_on=['PATIENT_ODU_ID','APPT_DATE'])

merged['any_revenue'] = merged['any_revenue'].fillna(False).astype(bool)
merged['any_inclinic'] = merged['any_inclinic'].fillna(False).astype(bool)
merged['total_amount'] = merged['total_amount'].fillna(0.0)
merged['ODU_IS_DELETED'] = merged['ODU_IS_DELETED'].fillna(False).astype(bool)

merged['no_show'] = ~( (merged['any_revenue']) & (merged['any_inclinic']) & (~merged['ODU_IS_DELETED']) )
print("no_show label created.")
print("Resulting DataFrame head:")
print(merged.head())

# Capitalize all column names before saving
print("\nCapitalizing all column names...")
merged.columns = [col.upper() for col in merged.columns]

# Final step: Save the result
output_file = 'analysis_results.csv'
merged.to_csv(output_file, index=False)
print(f"\nAnalysis complete. Results saved to '{output_file}'")
