import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the full trained model
model = tf.keras.models.load_model("next_word.h5")

# Get max sequence length used during training
max_sequence_length = model.input_shape[1] + 1

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_length - 1):]  # keep only relevant length
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return f"[Unknown index: {predicted_index}]"

# Streamlit UI
st.title("Next Word Prediction using LSTM (Shakespeare-Hamlet)")

input_text = st.text_input("Enter input text")

if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
        st.success(f"Predicted next word: **{next_word}**")
