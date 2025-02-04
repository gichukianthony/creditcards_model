import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

# Load the trained models and scaler
rf_model = joblib.load('model/random_forest.pkl')
xgb_model = joblib.load('model/xgboost.pkl')
scaler = joblib.load('model/scaler.pkl')

app = Flask(__name__)

# Home route (optional UI)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)

        rf_pred = rf_model.predict(features_scaled)[0]
        xgb_pred = xgb_model.predict(features_scaled)[0]

        return jsonify({"RandomForest_Prediction": int(rf_pred), "XGBoost_Prediction": int(xgb_pred)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
