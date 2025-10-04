"""from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/model.pkl')

# Root route
@app.route('/')
def home():
    return "Flight Price Prediction API is running! Use POST /predict with JSON data."

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[data['days_to_departure'], data['duration_mins'], data['total_stops']]])
        prediction = model.predict(features)[0]
        return jsonify({'predicted_price': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/model.pkl')

# Root route - show HTML form
@app.route('/')
def home():
    return render_template('index1.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        days_to_departure = int(request.form['days_to_departure'])
        duration_mins = int(request.form['duration_mins'])
        total_stops = int(request.form['total_stops'])
        
        features = np.array([[days_to_departure, duration_mins, total_stops]])
        prediction = model.predict(features)[0]
        
        return render_template('index1.html', prediction=round(float(prediction), 2))
    except Exception as e:
        return render_template('index1.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
