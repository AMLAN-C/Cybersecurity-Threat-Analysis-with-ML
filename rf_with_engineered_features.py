import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load data
df = pd.read_csv("CloudWatch_Traffic_Web_Attack.csv")

# Timestamp conversions
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['time'] = pd.to_datetime(df['time'])

# Feature engineering
df['session_duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
df['byte_ratio'] = df['bytes_out'] / (df['bytes_in'] + 1)

# New Feature 1: Time of Day
df['hour_of_day'] = df['time'].dt.hour

# New Feature 2: IP Frequency
df['src_ip_freq'] = df['src_ip'].map(df['src_ip'].value_counts())

# Final features list
features = ['bytes_in', 'bytes_out', 'session_duration', 'byte_ratio', 'hour_of_day', 'src_ip_freq']
X = df[features]

# Label anomalies using Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = iso.fit_predict(X)
df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
y = df['is_anomaly']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfc.fit(X_train, y_train)

# Predict probabilities & classes
y_proba = rfc.predict_proba(X_test)[:, 1]
threshold = 0.35
y_pred = (y_proba >= threshold).astype(int)

# Evaluation
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Optional: Feature importance plot
import matplotlib.pyplot as plt
import seaborn as sns

importances = rfc.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances After Engineering")
plt.tight_layout()
plt.show()
