from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model from the pickle file
with open('best_model_smote.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])
        
        # Prepare the feature array for prediction
        user_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
        
        # Make prediction
        prediction = model.predict(user_data)[0]
        result = "likely to have a stroke." if prediction == 1 else "not likely to have a stroke."
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
