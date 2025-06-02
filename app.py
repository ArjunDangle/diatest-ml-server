from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib # Or pickle, depending on how you saved your model

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing your React app to connect

# --- 1. Load your trained model ---
# Make sure this path is correct relative to your app.py
try:
    model = joblib.load('model.pkl') # <--- Adjust this path!
    
    # Get the feature names in the correct order as used during training
    feature_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []

# --- 2. Define the prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server setup."}), 500

    try:
        # Get the JSON data from the request
        data = request.json

        # Validate the request format as per the error message
        if not data or 'features' not in data or not isinstance(data['features'], list) or not data['features']:
            return jsonify({"error": "Invalid request format. Expected JSON with a non-empty 'features' list (e.g., {'features': [{...}]})."}), 400

        # Extract the single feature dictionary from the 'features' list
        input_data_dict = data['features'][0]

        # --- 3. Convert the dictionary to a DataFrame in the correct order ---
        # This is CRUCIAL for your model if it expects features in a specific order
        input_df = pd.DataFrame([input_data_dict], columns=feature_names)
        print("Backend: DataFrame for prediction:", input_df)

        # --- 4. Prepare input for prediction (no scaling applied) ---
        # Directly use the DataFrame's values as a NumPy array
        input_for_prediction = input_df.values
        

        # --- 5. Make the prediction ---
        prediction_class = int(model.predict(input_for_prediction)[0]) # Convert to int
        prediction_proba = model.predict_proba(input_for_prediction)[0].tolist() # Convert to list

        print(f"Backend: Prediction Class: {prediction_class}")
        print(f"Backend: Prediction Probabilities: {prediction_proba}")

        # --- 6. Return the prediction result ---
        return jsonify({
            "prediction_class": prediction_class,
            "prediction_proba": prediction_proba
        })

    except Exception as e:
        print(f"Error during prediction processing: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- 7. Run the Flask application ---
if __name__ == '__main__':
    app.run(debug=True, port=5001) # Set debug=True for development, port=5001