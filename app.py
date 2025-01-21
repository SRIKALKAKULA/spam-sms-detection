import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained model and vectorizer
tk = pickle.load(open("final_vectorizer.pkl", 'rb'))
model = pickle.load(open("final_best_model.pkl", 'rb'))

# Streamlit app
st.set_page_config(page_title="SMS Spam Detection", page_icon="📱", layout="centered")

# Adding custom CSS for styling
st.markdown("""
    <style>
        b{
            color: #ff6347
        }
        .title {
            font-size: 40px;
            color: #ff6347;
            font-weight: bold;
        }
        .subtitle {
            font-size: 12px;
            color: #FFFFFF;
            font-style: italic;
        }
        .input_box {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .button {
            background-color: #4caf50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# App UI with Emojis
st.markdown('<p class="title">📱 SMS Spam Detection Model</p>', unsafe_allow_html=True)


input_sms = st.text_input("Enter the SMS ✍️", key="input_sms", placeholder="Type your SMS here...", label_visibility="collapsed")


st.markdown('<p class="subtitle"><b>Spam Example:</b> Congratulations! You\'ve won a $1000 gift card! Claim it now by clicking on this link: [spamlink.com]. Hurry, offer expires soon!</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle"><b>Not Spam Example:</b> Hey, just wanted to check if we\'re still on for the meeting tomorrow at 3 PM. Let me know if that works for you!</p>', unsafe_allow_html=True)

if st.button('Predict 🔍', key='predict_button', help="Click to predict if the SMS is spam or not"):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tk.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result with emojis
    if result == 1:
        st.markdown('<h2 style="color: red;">🚨 Spam</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color: green;">✅ Not Spam</h2>', unsafe_allow_html=True)
        
        
st.markdown('<p class="subtitle">Made by Srikal Kakula ✨</p>', unsafe_allow_html=True)