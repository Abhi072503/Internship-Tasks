# firsttask.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas.api.types import is_string_dtype, is_object_dtype

# === 1. EXTRACT ===
df = pd.read_excel("pb-sales-data-blank.xlsx")  # Ensure this file is in the same folder
print("Original Data:\n", df.head())

# === 2. TRANSFORM ===

# Convert all datetime columns to strings (or handle them as needed)
for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
    df[col] = df[col].astype(str)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])  # Mode for categorical
    else:
        df[col] = df[col].fillna(df[col].mean())     # Mean for numeric

# Encode string/object columns
label_encoders = {}
for col in df.columns:
    if is_object_dtype(df[col]) or is_string_dtype(df[col]):
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        except Exception as e:
            print(f"⚠️ Skipping column '{col}' from Label Encoding due to: {e}")

# Normalize numeric columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n✅ Transformed Data:\n", df.head())

# === 3. LOAD ===
df.to_csv("cleaned_sales_data.csv", index=False)
print("\n✅ Cleaned data saved to 'cleaned_sales_data.csv'")
