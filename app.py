from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json

# Map to turn numbers to days/streets and vise versa
week_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
boro_map = {'Manhattan': 0, 'Bronx': 1, 'Brooklyn': 2, 'Queens': 3, 'Staten Island': 4}

# Load data
df = pd.read_csv('../data/Traffic_Volume_2024.csv')
street_list = df['street'].unique()
street_map = {name.strip().upper(): i for i, name in enumerate(street_list)}

# Create Flask App
app = Flask(__name__)

# Load trained model we made
model = joblib.load('model/traffic_model.pkl')

# Home route
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the NYC Traffic Volume Predictor API.', 'How to use': 'POST to /predict with JSON: weekday, hour, boro, street'})


# Define the route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data  = request.get_json()
    print("Received JSON:", data)
    try:
        # String inputs from user
        weekday = data.get('weekday')
        hour= int(data.get('hour'))
        boro = data.get('boro')
        street = data.get('street')

        # Covert from string input into our ML friendly int inputs
        weekday_num = week_map[weekday]
        boro_num = boro_map[boro]
        street_num = street_map[street.strip().upper()]

        input_array = [[weekday_num, hour, boro_num, street_num]]
        prediction = model.predict(input_array)
        return jsonify({'predicted_volume': int(prediction[0])})
    
    except KeyError as e:
        return jsonify({'error': f"Invalid input: {e.args[0]} not recognized"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)