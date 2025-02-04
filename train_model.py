import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('creditcard.csv')

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train RandomForest model
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost model
xgb_model = XGBClassifier(n_estimators=200, scale_pos_weight=round(y_train.value_counts()[0] / y_train.value_counts()[1]))
xgb_model.fit(X_train, y_train)

# Save models and scaler
joblib.dump(rf_model, 'model/random_forest.pkl')
joblib.dump(xgb_model, 'model/xgboost.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Models saved successfully in 'model/' folder.")
