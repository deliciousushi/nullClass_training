import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, Bidirectional, TimeDistributed, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import Embedding as KerasEmbedding

# 🚨 Custom Embedding class to ignore unsupported arguments
class CustomEmbedding(KerasEmbedding):
    def __init__(self, **kwargs):
        # Remove unsupported keys
        kwargs.pop('input_length', None)
        kwargs.pop('batch_input_shape', None)
        super().__init__(**kwargs)

# ✅ Load model with custom layers
model = load_model(
    'english_to_french_model.keras',
    custom_objects={
        'GRU': GRU,
        'Bidirectional': Bidirectional,
        'TimeDistributed': TimeDistributed,
        'Dense': Dense,
        'Dropout': Dropout,
        'Embedding': CustomEmbedding  # use the wrapper here
    }
)

# 📦 Load tokenizers
with open('english_tokenizer.json', 'r', encoding='utf8') as f:
    english_tokenizer = tokenizer_from_json(json.load(f))

with open('french_tokenizer.json', 'r', encoding='utf8') as f:
    french_tokenizer = tokenizer_from_json(json.load(f))

# 🔢 Load max sequence length
with open('sequence_length.json', 'r') as f:
    max_sequence_len = json.load(f)

# ✂️ Preprocess user input
def preprocess_input(sentence):
    sequence = english_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
    return padded

# 🔄 Convert output tokens to text
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words.get(np.argmax(vec), '') for vec in logits])

# 🌐 Streamlit UI
st.title("English to French Translator")
input_text = st.text_input("Enter an English sentence:")

if input_text:
    input_sequence = preprocess_input(input_text)
    prediction = model.predict(input_sequence)
    translation = logits_to_text(prediction[0], french_tokenizer)
    st.success(f"Predicted French translation: {translation.strip()}")

