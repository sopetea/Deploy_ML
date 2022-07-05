from sklearn.feature_extraction.text import CountVectorizer
import gzip
import numpy as np
import pickle
from flask import Flask, request, render_template, redirect
from PIL import Image


app = Flask(__name__)


@app.route('/')
def main():
    return redirect('/index')


@app.route('/index', methods=['GET'])
def index():
    return render_template('/index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        komentar = request.args.get('komentar')
    else:
        komentar = request.form['text']

    with open("model_covid.pkl", 'rb') as f:
        model = pickle.load(f)
    cv = CountVectorizer()
    data = [komentar]
    vect = cv.fit_transform(data).toarray()
    prediksi = model.predict([vect])
    proba = model.predict_proba([vect])
    if prediksi == 1:
        prediksi = "POSITIF"
        # emoji = Image.open("smile.png")
        # emoji = emoji.show()
        nega = "{:.0%}".format(proba[0][0])
        posi = "{:.0%}".format(proba[0][1])
    elif prediksi == -1:
        prediksi = "NEGATIF"
        # emoji = Image.open("angry.png")
        # emoji = emoji.show()
        nega = "{:.0%}".format(proba[0][0])
        posi = "{:.0%}".format(proba[0][1])
    else:
        print("Nothing")

    return (render_template('index.html', variable=prediksi, neg=nega, pos=posi))


@app.route('/about')
def about():
    return "Coba deploy simpel"


if __name__ == '__main__':
    app.run()
