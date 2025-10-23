
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('analysis_results.csv')

# --- 1. Basic Feature Engineering ---
print("\nStarting basic feature engineering...")

# Ensure date columns are parsed correctly
df["APPOINTMENT_DATETIME"] = pd.to_datetime(df["APPOINTMENT_DATETIME"], errors="coerce")
df["CREATED_DATE"] = pd.to_datetime(df["CREATED_DATE"], errors="coerce")

# Handle potential parsing errors
df.dropna(subset=["APPOINTMENT_DATETIME", "CREATED_DATE"], inplace=True)

# Create time-based features
df["LEAD_TIME_HOURS"] = (df["APPOINTMENT_DATETIME"] - df["CREATED_DATE"]).dt.total_seconds() / 3600
df["DAY_OF_WEEK"] = df["APPOINTMENT_DATETIME"].dt.day_name()
df["HOUR_OF_DAY"] = df["APPOINTMENT_DATETIME"].dt.hour
df["MONTH"] = df["APPOINTMENT_DATETIME"].dt.month
df["WEEKDAY_NUM"] = df["APPOINTMENT_DATETIME"].dt.weekday  # Monday=0, Sunday=6
df["IS_WEEKEND"] = df["WEEKDAY_NUM"].isin([5, 6])

print("Basic features created.")

# --- 2. Advanced Feature Engineering ---
print("\nStarting advanced feature engineering...")

# Appointment-related features
df["IS_MORNING_SLOT"] = df["HOUR_OF_DAY"].between(7, 11)
df["IS_AFTERNOON_SLOT"] = df["HOUR_OF_DAY"].between(12, 16)
df["IS_EVENING_SLOT"] = df["HOUR_OF_DAY"].between(17, 20)
df["DURATION_MIN"] = df["DURATION"]

# Patient behavior features
df = df.sort_values(["PATIENT_ODU_ID", "APPOINTMENT_DATETIME"])
df["PREV_APPT_DATE"] = df.groupby("PATIENT_ODU_ID")["APPOINTMENT_DATETIME"].shift(1)
df["DAYS_SINCE_LAST_APPT"] = (df["APPOINTMENT_DATETIME"] - df["PREV_APPT_DATE"]).dt.days

# Patient's past no-show rate (avoiding data leakage)
# The NO_SHOW column is boolean, so we convert to int for mean calculation
df["PAST_NOSHOW_RATE"] = (
    df.groupby("PATIENT_ODU_ID")["NO_SHOW"]
      .apply(lambda x: x.astype(int).shift().expanding().mean())
      .reset_index(level=0, drop=True)
)

# Clinic & staff consistency features
# Using transform is efficient for this calculation
df["RESOURCE_NOSHOW_RATE"] = (
    df.groupby("RESOURCE_ODU_ID")["NO_SHOW"].transform("mean")
)
df["PRACTICE_NOSHOW_RATE"] = (
    df.groupby("PRACTICE_ODU_ID")["NO_SHOW"].transform("mean")
)

# Time-window flags
df["IS_MONTH_START"] = df["APPOINTMENT_DATETIME"].dt.is_month_start
df["IS_MONTH_END"] = df["APPOINTMENT_DATETIME"].dt.is_month_end

print("Advanced features created.")

# --- 3. Exploratory Data Analysis (EDA) ---
print("\nStarting Exploratory Data Analysis (EDA)...")

# Missing values
print("\n--- Missing Values ---")
missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0])

# Summary stats
print("\n--- Summary Statistics ---")
summary_stats = df.describe(include="all")
print(summary_stats)

# Grouped no-show rates
print("\n--- Grouped No-Show Rates ---")
noshow_by_schedule = df.groupby("PIMS_SCHEDULE_TYPE", dropna=False)["NO_SHOW"].mean().reset_index()
print("\nBy Schedule Type:")
print(noshow_by_schedule)

noshow_by_day = df.groupby("DAY_OF_WEEK")["NO_SHOW"].mean().reset_index()
print("\nBy Day of Week:")
print(noshow_by_day)

noshow_by_hour = df.groupby("HOUR_OF_DAY")["NO_SHOW"].mean().reset_index()
print("\nBy Hour of Day:")
print(noshow_by_hour)

noshow_by_practice = df.groupby("PRACTICE_NAME")["NO_SHOW"].mean().reset_index()
print("\nBy Practice Name:")
print(noshow_by_practice)

# --- Visualizations ---
print("\nGenerating and saving visualizations...")

plt.figure(figsize=(10, 6))
sns.barplot(x="DAY_OF_WEEK", y="NO_SHOW", data=df,
            order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("No-show Rate by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/noshow_by_day.png")
print("Saved noshow_by_day.png")

plt.figure(figsize=(12, 6))
sns.barplot(x="HOUR_OF_DAY", y="NO_SHOW", data=df)
plt.title("No-show Rate by Hour of Day")
plt.tight_layout()
plt.savefig("output/noshow_by_hour.png")
print("Saved noshow_by_hour.png")

plt.figure(figsize=(10, 6))
sns.histplot(df["LEAD_TIME_HOURS"].clip(0, 1000), bins=50, kde=True) # Clip for better visualization
plt.title("Distribution of Lead Time (Hours, clipped at 1000)")
plt.tight_layout()
plt.savefig("output/lead_time_distribution.png")
print("Saved lead_time_distribution.png")

# --- Advanced Insights ---
print("\nCalculating advanced insights...")

# Correlation between lead time and no-show
corr = df["LEAD_TIME_HOURS"].corr(df["NO_SHOW"])
print(f"\nCorrelation between lead time and no-show: {corr:.3f}")

# Pivot heatmap: weekday vs hour
print("\nGenerating and saving heatmap...")
pivot = df.pivot_table(index="DAY_OF_WEEK", columns="HOUR_OF_DAY", values="NO_SHOW", aggfunc="mean")
plt.figure(figsize=(14, 8))
sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".2f")
plt.title("No-show Rate by Day and Hour")
plt.tight_layout()
plt.savefig("output/noshow_heatmap.png")
print("Saved noshow_heatmap.png")

# Save the final dataframe with all the new features
df.to_csv('output/final_features_and_eda.csv', index=False)
print("\nEDA complete. Final dataset with new features saved to 'output/final_features_and_eda.csv'.")
