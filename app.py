from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib

classifier = joblib.load(r'G:\MangeshDataScience\Practice\WorkEx\NLP\Email Spam Classification\EMSPMCLASS.pkl')
vectorizer = joblib.load(r'G:\MangeshDataScience\Practice\WorkEx\NLP\Email Spam Classification\vectorizer.pkl')
review = input(str('Enter the string: '))
tfidf = vectorizer.transform([review]).toarray()
pred = classifier.predict(tfidf)
if pred[0]==1:
    print('Spam')
else:
    print('Email')
