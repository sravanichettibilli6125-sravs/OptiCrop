import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("📈 Epic 4: Training Logistic Regression Model on Crop Data...")

# 1. Load data splits
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# 2. Train Logistic Regression
# max_iter is increased to 2000 because crop datasets have many classes and need more time to converge
lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)

# 3. Evaluate
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
print(f"📊 Logistic Regression Test Accuracy: {lr_acc * 100:.2f}%")

# 4. Save the model artifact
joblib.dump(lr_model, "models/model_lr.pkl")
print("💾 Logistic Regression model saved into models/model_lr.pkl!")