
from flask import Flask, request, render_template
import pandas as pd
import pickle
import sklearn

print(sklearn.__version__)

# Load the complete model pipeline
with open('best_crop_yield_model.pkl', 'rb') as file:
    dtr = pickle.load(file)

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        Year = float(request.form['Year'])
        average_rain_fall_mm_per_year = float(
            request.form['average_rain_fall_mm_per_year']
        )
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # Pass the original six features to the pipeline
        features = pd.DataFrame([{
            'Year': Year,
            'average_rain_fall_mm_per_year': average_rain_fall_mm_per_year,
            'pesticides_tonnes': pesticides_tonnes,
            'avg_temp': avg_temp,
            'Area': Area,
            'Item': Item
        }])

        # Pipeline handles preprocessing automatically
        prediction = dtr.predict(features)[0]

        return render_template(
            'index.html',
            prediction=round(float(prediction), 2)
        )


if __name__ == "__main__":
    app.run(debug=True)
