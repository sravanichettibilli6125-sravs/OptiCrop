import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("🤖 Epic 4: Training Random Forest Model on Crop Data...")

# 1. Load data splits
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# 2. Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 3. Evaluate
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"📊 Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

# 4. Save the model artifact
joblib.dump(rf_model, "models/model_rf.pkl")
print("💾 Random Forest model saved into models/model_rf.pkl!")