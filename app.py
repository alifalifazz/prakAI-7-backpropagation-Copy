from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan scaler
tf_scaler = joblib.load('scaler.save')
model = load_model('diabetes_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        try:
            # Ambil data dari form
            data = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age'])
            ]
            data_np = np.array(data).reshape(1, -1)
            data_scaled = tf_scaler.transform(data_np)
            pred = model.predict(data_scaled)[0][0]
            probability = float(pred) * 100
            prediction = 'Diabetes' if pred >= 0.5 else 'Tidak Diabetes'
        except Exception as e:
            prediction = f'Error: {e}'
    return render_template('index.html', prediction=prediction, probability=probability)

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)
