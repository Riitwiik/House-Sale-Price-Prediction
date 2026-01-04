from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load Model
with open('ridge_house_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    weights = model_data['weights']
    trained_features = model_data['features']

@app.route('/')
def home():
    # Pass an empty dictionary so 'original_input' is defined when the page first loads
    return render_template('index.html', original_input={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = request.form.to_dict()
        
        sqft = float(request.form.get('sqft_living', 0))
        grade = float(request.form.get('grade', 7))
        lat = float(request.form.get('lat', 47.6062))
        long = float(request.form.get('long', -122.3321))
        yr_built = float(request.form.get('yr_built', 2000))
        zipcode = int(request.form.get('zipcode', 98101))

        # Feature Engineering
        log_sqft = np.log(sqft) if sqft > 0 else 0
        data = {
            'bedrooms': float(request.form.get('bedrooms', 3)),
            'bathrooms': float(request.form.get('bathrooms', 2)),
            'log_sqft_living': log_sqft,
            'living_grade_interaction': log_sqft * grade,
            'dist_to_center': np.sqrt((lat - 47.6062)**2 + (long - (-122.3321))**2),
            'grade': grade,
            'waterfront': 0.0,
            'view': 0.0,
            'condition': 3.0,
            'house_age': 2015 - yr_built
        }

        # Handle Zipcodes
        X_df = pd.DataFrame([data])
        zip_cols = [f for f in trained_features if f.startswith('zip_')]
        zip_df = pd.DataFrame(0.0, index=[0], columns=zip_cols)
        if f"zip_{zipcode}" in zip_df.columns:
            zip_df[f"zip_{zipcode}"] = 1.0

        # Predict
        X_final = pd.concat([X_df, zip_df], axis=1)[trained_features]
        X_array = np.concatenate([np.ones((1, 1)), X_final.values], axis=1)
        prediction = np.exp(np.dot(X_array, weights))[0][0]

        return render_template('index.html', 
                               prediction_text=f'Estimated Price: ${prediction:,.2f}',
                               original_input=form_values)

    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error: {str(e)}', 
                               original_input=request.form.to_dict())

if __name__ == '__main__':
    app.run(debug=True)