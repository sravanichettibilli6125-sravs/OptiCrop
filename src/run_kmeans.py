import pandas as pd
from sklearn.cluster import KMeans
import joblib

print("🎯 Epic 3/4: Running K-Means Clustering on Crop Environments...")

# 1. Load the raw crop data measurements (dropping the label column)
df = pd.read_csv("data/raw/raw_dataset.csv")
X = df.drop(columns=['label'])

# 2. Find out how many unique crops are in the dataset to set our cluster count
num_classes = df['label'].nunique()
print(f"🌱 Found {num_classes} unique crop types in your dataset.")

# 3. Initialize and train K-Means to find that exact number of clusters
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
kmeans.fit(X)

# 4. Add the cluster assignments back to see what the AI decided
df['cluster_assignment'] = kmeans.labels_

# 5. Save the trained clustering model
joblib.dump(kmeans, "models/model_kmeans.pkl")
print("💾 K-Means model saved into models/model_kmeans.pkl!")

# 6. Show a quick sample of the environmental groupings
print("\n📊 Sample of K-Means groupings (First 5 rows):")
print(df[['N', 'P', 'K', 'temperature', 'cluster_assignment']].head())