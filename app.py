from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pickle
import pandas as pd
import os


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
	
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        with open('tf.sav', 'rb') as pickle_file:
            tfidf = pickle.load(pickle_file)
       # tfidf = pickle.load('tf.sav')
        features = tfidf.transform(data).toarray()
        with open('classifier.sav', 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        #model = pickle.load('classifier.sav')
        my_prediction = model.predict(features)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    pt = int(os.environ.get("PORT", 5000))
    app.run(host ='0.0.0.0',port=pt,debug=True,use_reloader=False) #host='127.0.0.1' port=pt
