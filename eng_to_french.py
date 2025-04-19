
import collections
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, GRU, LSTM, Bidirectional, Dropout, Activation, TimeDistributed, RepeatVector
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import streamlit as st

st.title("English to French Translation")

input_text = st.text_input("Enter an English sentence:")

if input_text:
    # Preprocess the input and get the prediction
    input_sequence = preprocess_input(input_text)
    prediction = model.predict(input_sequence)

    # Convert logits to text (translation)
    translated_text = logits_to_text(prediction, french_tokenizer)

    st.write(f"Predicted French translation: {translated_text}")

def load_data(path):
  input_file = path
  with open(input_file, "r") as f:
    data = f.read()
  return data.split('\n')

english_sentence = load_data('/content/small_vocab_en.csv')
french_sentence = load_data('/content/small_vocab_fr.csv')

english_sentence[1]
print(french_sentence[1])

english_word_counter = collections.Counter([word for sentence in english_sentence for word in sentence.split()])
french_word_counter = collections.Counter([word for sentence in french_sentence for word in sentence.split()])

print('{} English_words.'.format(len([word for sentence in english_sentence for word in sentence.split()])))
print('{} unique english words '.format(len(english_word_counter)))
print('10 most commomn words:')
print('"' + '" "'.join(list(zip(*english_word_counter.most_common(10)))[0]) + '"')
print()

print('{} French_words.'.format(len([word for sentence in french_sentence for word in sentence.split()])))
print("{} unique French words.".format(len(french_word_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_word_counter.most_common (10)))[0]) + '"')

english_word_counter = collections.Counter([word for sentence in english_sentence for word in sentence.split()])
french_word_counter = collections.Counter([word for sentence in french_sentence for word in sentence.split()])

# Corrected print statements
print('{} English words'.format(len([word for sentence in english_sentence for word in sentence.split()])))
print('{} Unique English words'.format(len(english_word_counter)))
print('10 most common English words:', english_word_counter.most_common(10))

print('{} French words'.format(len([word for sentence in french_sentence for word in sentence.split()])))
print('{} Unique French words'.format(len(french_word_counter)))
print('10 most common French words:', french_word_counter.most_common(10))

def tokenize(x):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(x)
  return tokenizer.texts_to_sequences(x), tokenizer

text_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "By Jove, my quick study of lexicography won a prize.",
    "This is a short sentence."]
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)

print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print(' Sequence {} in x'.format(sample_i + 1))
    print(' Input: {}'.format(sent))
    print(' Output: {}'.format(token_sent))

def pad(x, length=None):
  if length is None:
    length = max([len(sentence) for sentence in x])
  return pad_sequences (x, maxlen=length, padding='post')

test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
  print('Sequence {} in x'.format(sample_i + 1))
  print(' Input: {}'.format(np.array(token_sent)))
  print(' Output: {}'.format(pad_sent))

def preprocess (x,y):
  preprocess_x, x_tk = tokenize(x)
  preprocess_y, y_tk = tokenize(y)
  preprocess_X = pad(preprocess_x)
  preprocess_y = pad(preprocess_y)
  preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

  return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentence, french_sentence)
#preproc_english_sentences = np.array(preproc_english_sentences)
#preproc_french_sentences = np.array(preproc_french_sentences)

english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

max_english_sequence_length = max(len(seq) for seq in preproc_english_sentences)
max_french_sequence_length = max(len(seq) for seq in preproc_french_sentences)

# Pad sequences
preproc_english_sentences = pad(preproc_english_sentences, length=max_english_sequence_length)
preproc_french_sentences = pad(preproc_french_sentences, length=max_french_sequence_length)

# Convert lists to NumPy arrays
preproc_english_sentences = np.array(preproc_english_sentences)
preproc_french_sentences = np.array(preproc_french_sentences)

english_vocab_size = len(english_tokenizer.word_index) + 1  # Add 1 for padding token
french_vocab_size = len(french_tokenizer.word_index) + 1  # Add 1 for padding token

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

def logits_to_text (logits, tokenizer):
  index_to_words = {id: word for word, id in tokenizer.word_index.items()}
  index_to_words[0] = '<PAD>'

  return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

"""**SIMPLE MODEL**"""

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  learning_rate = 0.005
  model = Sequential()

  model.add(GRU(256, input_shape = input_shape[1:], return_sequences = True))
  model.add(TimeDistributed(Dense(1024, activation='relu')))
  model.add(Dropout(0.5))
  model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

  model.compile(loss = sparse_categorical_crossentropy,
                optimizer = Adam(learning_rate),
                metrics = ['accuracy'])
  return model

tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)

simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1834, epochs=10, validation_split=0.2)

print("Prediciton:")
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

print("\nCorrect Translation:")
print(french_sentence[:1])

print("\n original text:")
print(english_sentence[:1])

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  learning_rate = 0.005
  model = Sequential()
  model.add(Bidirectional (GRU(128, return_sequences=True), input_shape=input_shape[1:]))
  model.add(TimeDistributed(Dense(1024, activation='relu')))
  model.add(Dropout(0.5))
  model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))

  model.compile(loss = sparse_categorical_crossentropy,
                optimizer = Adam(learning_rate),
                metrics = ['accuracy'])
  return model

tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

bd_rnn_model = bd_model(
    tmp_x.shape, max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)

print(bd_rnn_model.summary())
bd_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

print("Prediciton:")
print(logits_to_text(bd_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

print("\nCorrect Translation:")
print(french_sentence[:1])

print("\n original text:")
print(english_sentence[:1])

def bidirectional_embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  learning_rate = 0.005
  model = Sequential()
  model.add(Embedding(english_vocab_size, 256, input_length=input_shape[1], input_shape=input_shape[1:]))
  model.add(Bidirectional (GRU(256, return_sequences=True)))
  model.add(TimeDistributed(Dense(1024, activation='relu')))
  model.add(Dropout(0.5))
  model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

  model.compile(loss=sparse_categorical_crossentropy,
                optimizer=Adam(learning_rate),
                metrics=['accuracy'])

  return model

tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

embed_rnn_model = bidirectional_embed_model( tmp_x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)
print(embed_rnn_model.summary())
embed_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

print("Prediciton:")
print(logits_to_text(embed_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

print("\nCorrect Translation:")
print(french_sentence[:1])

print("\n original text:")
print(english_sentence[:1])

embed_rnn_model.save('english_to_french_model')

#serialize English Tokenizer to JSON
with open('english_tokenizer.json', 'w', encoding='utf8') as f:
      f.write(json.dumps(english_tokenizer.to_json(), ensure_ascii=False))

#Serialize French Tokenizer to JSON
with open('french_tokenizer.json', 'w', encoding='utf8') as f:
      f.write(json.dumps(french_tokenizer.to_json(), ensure_ascii=False))

#Save max Lengths
max_french_sequence_length_json = max_french_sequence_length
with open('sequence_length.json', 'w', encoding='utf8') as f:
      f.write(json.dumps(max_french_sequence_length_json, ensure_ascii=False))
