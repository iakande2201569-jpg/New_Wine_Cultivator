from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Robust absolute path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'wine_cultivar_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    inputs = {}
    if request.method == 'POST':
        try:
            inputs = request.form.to_dict()
            data = {
                'alcohol': [float(request.form['alcohol'])],
                'malic_acid': [float(request.form['malic_acid'])],
                'magnesium': [float(request.form['magnesium'])],
                'total_phenols': [float(request.form['total_phenols'])],
                'flavanoids': [float(request.form['flavanoids'])],
                'color_intensity': [float(request.form['color_intensity'])]
            }
            prediction = model.predict(pd.DataFrame(data))[0]
            # Mapping target classes to Cultivar names
            result = f"Cultivar {prediction + 1}"
        except Exception:
            # User-friendly error message
            result = "Error: Please check your inputs and try again."

    return render_template('index.html', prediction=result, inputs=inputs)

if __name__ == "__main__":
    # Debug mode disabled for hosted environment
    app.run(debug=False)