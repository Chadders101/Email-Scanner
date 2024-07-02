from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = load('final.joblib')

# Define the model features in the order they were used for training
model_features = [
    "nb_links", "length_url", "length_hostname", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", 
    "nb_and", "nb_eq", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon", 
    "nb_comma", "nb_semicolon", "nb_dollar", "nb_www", "nb_com", "nb_dslash", 
    "ratio_digits_url"
]

@app.route('/test', methods=['POST'])
def predict_phishing():
    try:
        # Parse the incoming JSON data
        data = request.json['linksData']
        model_faetures = ["nb_links"]
        #logging.info(f"Received data: {data}")
        
        # Ensure the data is in the correct order as expected by the model
        ordered_data = {feature: data.get(feature, 0) for feature in model_faetures}
        #prediction_proba = data.get('nb_links')
        
        # Convert the ordered data into the DataFrame
        features_df = pd.DataFrame([ordered_data], columns=model_faetures)
        logging.info("DataFrame created from received data:")
        #logging.info(features_df.describe(include='all'))
        
        # Use the model to predict
        prediction_proba = model.predict_proba(features_df)[0][1]  # Probability of being a phishing attempt
        
        # Decide on the colour based on the prediction probability
        colour = "#00ff00"  # Default to Low probability
        if prediction_proba >= 0.8:
            colour = "#ff0000"  # High probability of phishing
        elif prediction_proba >= 0.5:
            colour = "#ffa500"  # Medium probability
        
        return jsonify({"colour": colour})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Error processing request"}), 500

if __name__ == '__main__':
    app.run(debug=True)
