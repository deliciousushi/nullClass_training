import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load model
model = load_model('english_to_french_model.keras')

# Load tokenizers
with open('english_tokenizer.json', 'r', encoding='utf8') as f:
    english_tokenizer = tokenizer_from_json(json.load(f))

with open('french_tokenizer.json', 'r', encoding='utf8') as f:
    french_tokenizer = tokenizer_from_json(json.load(f))

# Load max sequence length
with open('sequence_length.json', 'r', encoding='utf8') as f:
    max_french_sequence_length = json.load(f)

# Utility to convert logits to words
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words.get(prediction, '') for prediction in np.argmax(logits, axis=1)])

# Preprocess input
def preprocess_input(sentence):
    sequence = english_tokenizer.texts_to_sequences([sentence.lower()])
    padded = pad_sequences(sequence, maxlen=max_french_sequence_length, padding='post')
    return padded

# Streamlit UI
st.title("English to French Translator 🗣️➡️🗼")
st.write("Type an English sentence and get the French translation.")

input_text = st.text_input("Enter an English sentence:")

if input_text:
    input_seq = preprocess_input(input_text)
    prediction = model.predict(input_seq)
    translated = logits_to_text(prediction[0], french_tokenizer)

    st.success(f"**French translation:** {translated}")
