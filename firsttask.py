
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load dataset
df = pd.read_csv("sample_data.csv")  # Make sure this file is in the same folder

# Step 2: Clean data
df.dropna(inplace=True)             # Remove rows with missing values
df.drop_duplicates(inplace=True)    # Remove duplicate rows

# Step 3: Encode categorical columns (like Gender, Department)
le = LabelEncoder()
if 'Gender' in df.columns:
    df['Gender'] = le.fit_transform(df['Gender'])
if 'Department' in df.columns:
    df['Department'] = le.fit_transform(df['Department'])

# Step 4: Scale numeric columns (like Age, Salary)
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 5: Save the cleaned and processed data
df.to_csv("cleaned_data.csv", index=False)

print("✅ Data pipeline completed! File saved as cleaned_data.csv.")