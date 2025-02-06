import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load datasetimport pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train RandomForest model with reduced overfitting
rf_model = RandomForestClassifier(
    n_estimators=100,   # Reduce from 200 to 100
    max_depth=10,       # Limit tree depth
    class_weight='balanced', 
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

# Train XGBoost model with higher fraud weight
xgb_model = XGBClassifier(
    n_estimators=200, 
    scale_pos_weight=10  # Increase fraud importance
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Save models and scaler
joblib.dump(rf_model, 'model/random_forest.pkl')
joblib.dump(xgb_model, 'model/xgboost.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Evaluate Models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for fraud (Class 1)
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Adjust threshold
    threshold = 0.2  # Lower threshold for fraud detection
    y_pred_adjusted = (y_prob >= threshold).astype(int)
    
    print(f"{model_name} Adjusted Classification Report (Threshold = {threshold}):")
    print(classification_report(y_test, y_pred_adjusted))
    print(f"{model_name} Adjusted Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_adjusted))
    print("\n--------------------------------------------------\n")

# Run evaluations
evaluate_model(rf_model, X_test, y_test, "RandomForest")
evaluate_model(xgb_model, X_test, y_test, "XGBoost")

print("Models saved successfully in 'model/' folder.")

data = pd.read_csv('creditcard.csv')

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train RandomForest model with reduced overfitting
rf_model = RandomForestClassifier(
    n_estimators=100,   # Reduce from 200 to 100
    max_depth=10,       # Limit tree depth
    class_weight='balanced', 
    random_state=42
)
rf_model.fit(X_train, y_train)

# Train XGBoost model with higher fraud weight
xgb_model = XGBClassifier(
    n_estimators=200, 
    scale_pos_weight=10  # Increase fraud importance
)
xgb_model.fit(X_train, y_train)

# Save models and scaler
joblib.dump(rf_model, 'model/random_forest.pkl')
joblib.dump(xgb_model, 'model/xgboost.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Models saved successfully in 'model/' folder.")
