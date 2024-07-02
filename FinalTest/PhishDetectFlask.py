from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = load('rf_model.joblib')

@app.route('/test', methods=['POST'])
def predict_phishing():
    try:
        # Parse the incoming JSON data
        data = request.json['linksData']
        logging.info(f"Received data: {data}")
        
        # Convert the data into the format expected by the model
        features_df = pd.DataFrame([data])
        logging.info("DataFrame created from received data:")
        logging.info(features_df.describe(include='all'))
        
        # Use the model to predict
        prediction_proba = model.predict_proba(features_df)[0][1]  # Probability of being a phishing attempt
        
        # Decide on the color based on the prediction probability
        if prediction_proba >= 0.8:
            color = "red"  # High probability of phishing
        elif prediction_proba >= 0.5:
            color = "yellow"  # Medium probability
        else:
            color = "green"  # Low probability
        
        return jsonify({"color": color})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Error processing request"}), 500

if __name__ == '__main__':
    app.run(debug=True)
