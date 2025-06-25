# LSTM_Word_Predictor

# 🧠 Next Word Prediction using LSTM

An interactive web application that predicts the next word in a sentence using a deep learning model built with LSTM (Long Short-Term Memory) layers. This project is implemented in TensorFlow and deployed using Streamlit.

---

## 🚀 Demo

Input a short phrase like:

```
To be or not to
```

Click **Predict Next Word** and see the model's prediction based on the training data!

---

## 📚 Project Description

This project demonstrates how to use Recurrent Neural Networks (specifically LSTM) for next-word prediction — a fundamental task in Natural Language Processing (NLP).

- **Dataset**: Shakespeare's *Hamlet*
- **Model**: Embedding → LSTM → Dense
- **Frontend**: Built using Streamlit for easy user interaction
- **Tokenizer**: Word-level tokenization using Keras

---

## 📁 Project Structure

```
LSTM_RNN/
├── app.py                # Streamlit app (main interface)
├── experiments.ipynb     # Notebook for training and experimentation
├── next_word.h5          # Trained LSTM model weights
├── tokenizer.pickle      # Saved tokenizer for word-index mapping
├── hamlet.txt            # Text corpus used for training
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/lstm-next-word-predictor.git
cd lstm-next-word-predictor
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
# Activate it:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Details

The model is a sequential Keras model with:

- **Embedding Layer**: Converts token indices to dense vectors
- **LSTM Layer**: Captures sequential dependencies
- **Dense Output Layer**: Predicts probabilities across the vocabulary

---

## 💡 Training

Model training is done in `experiments.ipynb`. It reads `hamlet.txt`, prepares sequences, tokenizes the text, and trains the model.

Once trained, the model is saved to `next_word.h5`, and the tokenizer to `tokenizer.pickle`.

---

## 📝 Requirements

- TensorFlow==2.10.0
- Keras==2.10.0
- Streamlit
- pandas
- numpy
- nltk
- matplotlib
- scikit-learn

Install using:
```bash
pip install -r requirements.txt
```

---

## 🙋 Author

**Mayank Rathi**  
🔗 [GitHub: @rmayank-24](https://github.com/rmayank-24)

---

## 🪪 License

This project is licensed under the MIT License.
