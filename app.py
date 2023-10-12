from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the saved models and encoders
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        user_input = np.array[[N, P, K, temperature, humidity, ph, rainfall]]  # Include rainfall in the user input
        user_input = model.predict(user_input)

    # Map the predicted label back to the original crop name
        crop_mapping = {code: crop for code, crop in enumerate(crop['crop'].cat.categories)}
        predicted_crop = crop_mapping[user_input[0]]
        
        result = predicted_crop
    return render_template('index.html',result=result )

if __name__ == '__main__':
    app.run(debug=True)
   
