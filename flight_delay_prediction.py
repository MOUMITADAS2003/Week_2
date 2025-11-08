# ===============================
# Aircraft Delay Prediction System 🛫
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv('flights.csv', low_memory=False)
print("✅ Dataset Loaded Successfully!")
print("Shape of data:", df.shape)
print(df.head())

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------
# Select important columns
columns_to_use = ['Origin', 'Dest', 'DepDelayMinutes',
                  'ArrDelayMinutes', 'Distance', 'DayOfWeek']

df = df[columns_to_use]

# Create target variable (1 = Delayed, 0 = On Time)
df['Delayed'] = np.where(df['ArrDelayMinutes'] > 15, 1, 0)
df.drop('ArrDelayMinutes', axis=1, inplace=True)

# Fill missing values
df = df.fillna(0)

# Encode categorical columns
label_encoders = {}
for col in ['Origin', 'Dest']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------------
# Step 3: Split Data
# -------------------------------
X = df.drop('Delayed', axis=1)
y = df['Delayed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 6: Feature Importance Visualization
# -------------------------------
importances = model.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Importance in Flight Delay Prediction', fontsize=14, weight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()
