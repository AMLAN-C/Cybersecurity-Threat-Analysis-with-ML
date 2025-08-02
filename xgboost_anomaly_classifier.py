import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("CloudWatch_Traffic_Web_Attack.csv")
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['time'] = pd.to_datetime(df['time'])

# Feature Engineering
df['session_duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
df['byte_ratio'] = df['bytes_out'] / (df['bytes_in'] + 1)
df['hour_of_day'] = df['time'].dt.hour
df['src_ip_freq'] = df['src_ip'].map(df['src_ip'].value_counts())

# Feature Set
features = ['bytes_in', 'bytes_out', 'session_duration', 'byte_ratio', 'hour_of_day', 'src_ip_freq']
X = df[features]

# Label with Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = iso.fit_predict(X)
df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
y = df['is_anomaly']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight
n_pos = sum(y_train == 1)
n_neg = sum(y_train == 0)
scale_pos_weight = n_neg / n_pos

# XGBoost Classifier
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)

# Predictions
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

# Evaluation
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Feature Importance
importances = xgb.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()
