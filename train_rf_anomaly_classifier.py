import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("CloudWatch_Traffic_Web_Attack.csv")

# Convert timestamps
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['time'] = pd.to_datetime(df['time'])

# Feature Engineering
df['session_duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
df['byte_ratio'] = df['bytes_out'] / (df['bytes_in'] + 1)

# Select features for modeling
features = ['bytes_in', 'bytes_out', 'session_duration', 'byte_ratio']
X = df[features]

# Use Isolation Forest to label anomalies
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = iso.fit_predict(X)
df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
y = df['is_anomaly']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predict
y_pred = rfc.predict(X_test)

# Evaluate
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Optional: Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

importances = rfc.feature_importances_
feat_names = features
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance Score")
plt.show()
