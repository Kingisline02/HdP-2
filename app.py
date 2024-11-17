from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the data from pickle file
pkl_filename = "data.pkl"
df = pd.read_pickle(pkl_filename)

# Extract features and target
X = df[["Age", "Height", "Weight", "Blood Pressure", "Smoking Status"]].values
y = df["Target"].astype(int).values

# Compute BMI
heights_in_m = X[:, 1] / 100
bmis = X[:, 2] / (heights_in_m ** 2)
X = np.column_stack((X, bmis))

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the model and scaler from pickle files
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    bp = int(request.form['bp'])
    smoking = int(request.form['smoking'])

    # Compute BMI
    bmi = weight / ((height / 100) ** 2)

    # Create feature array
    features = np.array([[age, height, weight, bp, smoking, bmi]])
    features_scaled = scaler.transform(features)

    # Predict the risk
    prediction = model.predict(features_scaled)[0]
    risk_level = ['Low', 'Moderate', 'High'][int(prediction)]

    # Define the color based on the risk level
    colors = {
        'Low': 'green',
        'Moderate': 'yellow',
        'High': 'red'
    }
    risk_color = colors[risk_level]

    return render_template('index.html', risk_level=risk_level, risk_color=risk_color)

if __name__ == '__main__':
    app.run(debug=True)
