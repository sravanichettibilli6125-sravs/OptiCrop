import pandas as pd
import os

print("📥 Epic 2: Processing your custom Crop Recommendation dataset...")

# Explicit path check
source_file = "Crop_recommendation.csv"

if os.path.exists(source_file):
    df = pd.read_csv(source_file)
    os.makedirs("data/raw", exist_ok=True)
    
    # Save directly over the old file
    df.to_csv("data/raw/raw_dataset.csv", index=False)
    print("💾 Success! Overwrote old data with Crop dataset.")
    print(f"📊 Verified columns inside data/raw/raw_dataset.csv: {list(df.columns)}")
else:
    print(f"❌ Error: Could not find '{source_file}' in your main project folder.")
    print("Please make sure the file is dropped directly into the 'Opticrop' folder!")