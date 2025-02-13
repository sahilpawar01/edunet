from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and dummy columns
model = pickle.load(open('energy_model.pkl', 'rb'))
dummy_columns = pickle.load(open('dummy_columns.pkl', 'rb'))

def predict_energy_consumption(year, country):
    # Prepare the input DataFrame.
    input_df = pd.DataFrame({'Year': [year], 'Country': [country]})
    input_encoded = pd.get_dummies(input_df, columns=['Country'])
    input_encoded = input_encoded.reindex(columns=dummy_columns, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    return prediction

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction_text=None, country=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form inputs.
    year = int(request.form.get('year'))
    country = request.form.get('country')
    
    # Make prediction.
    prediction = predict_energy_consumption(year, country)
    # Include units in the displayed text (modify "Units" to your actual unit).
    prediction_text = f'Predicted Energy Consumption for {country} in {year}: {prediction:.2f} Units'
    
    # Pass the country and numeric prediction value for the map.
    return render_template('index.html', prediction_text=prediction_text, country=country, prediction_value=prediction)

if __name__ == '__main__':
    app.run(debug=True)
