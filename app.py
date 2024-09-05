from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

with open('models/logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']

label_encoders = {} 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': float(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges'])
    }

    df = pd.DataFrame([data])

    for col in categorical_columns:
        if col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:  # Binary columns
            df[col] = df[col].map({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0})
        elif col not in label_encoders and df[col].nunique() > 2:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    df_processed = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction_prob = model.predict_proba(df_processed)[:, 1]
    churn_probability = round(prediction_prob[0], 2) * 100

    return render_template('index.html', prediction_text=f'Churn Probability: {churn_probability}%')

if __name__ == "__main__":
    app.run(debug=True)
