from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pickle
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_excel('C:\\Users\\Ashish Pawar\\Desktop\\HeavyWater\\shuffled-full-set-hashed.xlsx')
    df=df[pd.notnull(df['content'])]
    df['Tag_id']=df['Tags'].factorize()[0]
    tag_id_df = df[['Tags','Tag_id']].drop_duplicates().sort_values('Tag_id')
    tag_to_id = dict(tag_id_df.values)
    id_to_tag = dict(tag_id_df[['Tag_id','Tags']].values)
    tfidf = TfidfVectorizer(min_df=0.2)
    features = tfidf.fit_transform(df.content).toarray()
    labels = df.Tag_id
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25,random_state=27)
    sm = SMOTE('minority',random_state=27)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    logreg=LogisticRegression()
    logreg.fit(X_train,y_train)
    
	
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        features = tfidf.transform(data).toarray()
        my_prediction = logreg.predict(features)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)