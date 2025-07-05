import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import string

def transform_text(text):
    text = text.lower() # lowercase

    text = nltk.word_tokenize(text) # Tokenization
    y = [] # removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    # removing stop words and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    if input_sms:
        # Transform the input text
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict using the model
        result = model.predict(vector_input)[0]
        
        # Display the result
        if result == 1:
            st.header("Spam Message 🚫")
        else:
            st.header("Ham (Not Spam) Message ✅")
    else:
        st.warning("Please enter a message to classify.")

