import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("🧹 Epic 3: Cleaning Crop data and preparing splits...")
df = pd.read_csv("data/raw/raw_dataset.csv")

# Strip any accidental hidden spaces from the column names
df.columns = df.columns.str.strip()

# Automatically find the label column (handling 'label', 'Label', etc.)
target_col = [col for col in df.columns if col.lower() == 'label']

if not target_col:
    print(f"❌ Error: Could not find a label column. Your dataset columns are: {list(df.columns)}")
    exit()

actual_target = target_col[0]

# 1. Translate crop labels (text) into numbers
encoder = LabelEncoder()
df[actual_target] = encoder.fit_transform(df[actual_target])

# 2. Save this translator so our web app can map numbers back to crop names later
os.makedirs("models", exist_ok=True)
joblib.dump(encoder, "models/label_encoder.pkl")

# 3. Separate features (soil/weather metrics) from the target (crop label)
X = df.drop(columns=[actual_target])
y = df[actual_target]

# 4. Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Save everything into the data/processed folder
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("✅ Done! Processed crop data files created in data/processed/")