from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('mlp_model.joblib')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        frequency = float(request.form['frequency'])
        # Preprocess the input data
        input_data = np.array([[frequency]])
        # Perform inference using the loaded model
        prediction = model.predict(input_data)
        # Map the prediction to a human-readable string
        if prediction == 1:
            result = "Quality is low"
            result_class = "low"
        elif prediction == 2:
            result = "Quality is medium"
            result_class = "medium"
        elif prediction == 0:
            result = "Quality is high"
            result_class = "high"
        else:
            result = "Unknown"
            result_class = ""

        return render_template('main.html', result=result, result_class=result_class)
    except Exception as e:
        return render_template('main.html', result=f"Error: {str(e)}", result_class="")

if __name__ == '__main__':
    app.run(debug=True)
