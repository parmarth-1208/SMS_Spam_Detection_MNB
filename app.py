import streamlit as st
import pickle
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)
    
    y = []
    for i in message:
        if i.isalnum():
            y.append(i)
            
    message = y[:]
    y.clear()
    
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    message = y[:]
    y.clear()
    
    for i in message:
        y.append(ps.stem(i))
            
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS SPAM DETECTOR")

input_sms = st.text_area("Enter the message: ")


if st.button('PREDICT'):
    transform_sms = transform_message(input_sms)
    
    vector_input = tfidf.transform([transform_sms])
    
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("SPAM")
        
    else:
        st.header("NOT SPAM")