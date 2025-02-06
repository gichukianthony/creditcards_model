# Fraud Detection API

## Overview
The **Fraud Detection API** is a machine learning-powered API designed to detect fraudulent activities based on input features. It utilizes **RandomForest** and **XGBoost** models to predict whether a transaction is fraudulent or not.

## Features
✅ Predict fraudulent transactions using machine learning  
✅ Supports multiple classification models (RandomForest, XGBoost)  
✅ Simple web interface for user input and visualization  
✅ Well-structured API with JSON responses  

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript (Vanilla JS)  
- **Backend:** Python (Flask/FastAPI)  
- **Machine Learning Models:** RandomForest, XGBoost  
- **Styling:** Tailwind CSS  

## Installation & Setup

1. **Clone the repository**  
   ```sh
   git clone https://github.com/yourusername/fraud-detection-api.git
   cd fraud-detection-api

 2.**install the dependancies**
pip install -r requirements.txt
 
 3. **run the api server**
python app.py  # Flask-based backend

4. **Run the frontend**
Open index.html in your browser.

API Endpoints
1. Predict Fraud
URL: /predict
Method: POST
Headers: Content-Type: application/json
Request Body:
json
Copy
Edit


request body
{
  "features": [10.5, -7.2, 15.0, -3.0, 20.4, -5.5, 30.0]
}

json response
{
  "RandomForest_Prediction": 0,
  "XGBoost_Prediction": 1
}

