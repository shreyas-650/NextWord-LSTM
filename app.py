import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

model = load_model('nextword_model.h5')

with open('tok.pkl','rb') as file:
    tok = pickle.load(file)

reversed_index = {idx:word for word,idx in tok.word_index.items()}

max_len = 44

def generate_text(seed_text,num_words=10):
    text = seed_text
    for _ in range(num_words):
        seq = tok.texts_to_sequences([text])[0]
        padded = pad_sequences([seq],padding='pre',maxlen=max_len)
        preds = model.predict(padded,verbose=0)
        pos = np.argmax(preds)
        next_word = reversed_index.get(pos," ")
        text += " "+next_word
    return text

st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="💬",
    layout="centered"
)

st.title('Next Word Prediction with Deep Learning')
st.subheader("LSTM Based Language Model")
seed=st.text_input('Enter a Starting text: ','Hello')
num_words = st.slider('Number of words to generate',1,100,10)

if st.button('Generate'):
    result = generate_text(seed,num_words)
    st.write(result)