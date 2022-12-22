from crypt import methods
import imp
from pyexpat import features
import pandas as pd


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
svm = pickle.load(open("svm.pkl", "rb"))

data = pd.read_csv('./data/cellphones data.csv')
data = data.drop(['operating system', 'price'], axis = 1)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    operating_system = request.form.get("operating system")
    occupation = request.form.get("occupation")
    age = request.form.get("age")
    gender = request.form.get("gender")
    rating = request.form.get("rating")
    price = request.form.get("price")

    df = pd.DataFrame({'rating' : rating, 'operating system' : operating_system, 'price' : price, 'gender' : gender, 'age' : age, 'occupation' : occupation}, index=[0])

    object_data = ['operating system', 'occupation']

    all_objects = []
    for rows in df[object_data].fillna(" ").values.tolist():
        for phrases in rows:
            all_objects.append(phrases)

    all_objects = list(set(all_objects))

    def alphabet_to_number(x):
        try:
            return all_objects.index(x)
        except:
            return 404 # error not found

    def gender_null(gender_clear):
        try:
            if str(gender_clear).lower() == 'female' :
                return 0
            elif str(gender_clear).lower() == 'male' :
                return 1
        except:
            return 2
    
    df['operating system'] = df['operating system'].apply(alphabet_to_number)
    df['occupation'] = df['occupation'].apply(alphabet_to_number)
    df['gender'] = df['gender'].apply(gender_null)

    df = df.astype(np.float32)
    
    prediction = svm.predict(df)
    df['cellphone_id'] = prediction
    df = df.set_index('cellphone_id').join(data)
    return render_template("prediksi.html",  
    brand_name = '{}'.format(df.iloc[0]['brand']),
    model_name = '{}'.format(df.iloc[0]['model']),
    internal_memory = '{}'.format(df.iloc[0]['internal memory']),
    RAM = '{}'.format(df.iloc[0]['RAM']),
    main_camera = '{}'.format(df.iloc[0]['main camera']),
    selfie_camera = '{}'.format(df.iloc[0]['selfie camera']),
    battery_size = '{}'.format(df.iloc[0]['battery size']),
    screen_size = '{}'.format(df.iloc[0]['screen size']),
    weight = '{}'.format(df.iloc[0]['weight']),
    release_date = '{}'.format(df.iloc[0]['release date']),
    )

if __name__ == "__main__":
    app.run(debug=True)