from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and metadata
model = pickle.load(open('model.pkl','rb'))
columns = pickle.load(open('columns.pkl','rb'))
options = pickle.load(open('options.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Convert numeric input
    data['Horsepower'] = float(data['Horsepower'])

    # Convert Year to string
    data['Year'] = str(data['Year'])

    # DataFrame
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure same order
    df = df[columns]

    # Predict
    prediction = model.predict(df)[0]

    return render_template('index.html', prediction_text=f'Predicted Price: ₹ {round(prediction,2)}', options=options)

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)