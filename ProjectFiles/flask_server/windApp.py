# Import required libraries
import numpy as np
from flask import Flask, request, render_template
import joblib
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -------------------------------
# Initialize the Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load the model
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), '..', 'power_prediction.sav')
model = joblib.load(model_path)

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/windapi', methods=['POST'])
def windapi():
    city = request.form.get('city')
    apikey = os.getenv('OPENWEATHER_API_KEY')

    if not apikey:
        return render_template('predict.html', error_msg="OpenWeather API key not set.")

    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}'

    try:
        resp = requests.get(url, timeout=10).json()
        if resp.get('cod') != 200:
            return render_template('predict.html', error_msg=f"City not found: {city}")

        temp = str(round(resp['main']['temp'] - 273.15, 1)) + ' °C'
        humid = str(resp['main']['humidity']) + ' %'
        pressure = str(resp['main']['pressure']) + ' hPa'
        speed = str(resp['wind']['speed']) + ' m/s'

    except Exception as e:
        return render_template('predict.html', error_msg=f"Failed to fetch weather data: {e}")

    return render_template('predict.html', temp=temp, humid=humid, pressure=pressure, speed=speed)

@app.route('/y_predict', methods=['POST'])
def y_predict():
    val_X = [[float(x) for x in request.form.values()]]
    prediction = model.predict(val_X)
    output = prediction[0]
    return render_template('predict.html', prediction_text=f'The energy predicted is {output:.2f} KWh')

# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
