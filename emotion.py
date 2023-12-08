import streamlit as st
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model and label encoder
with open('lrc_vsfc.pkl', 'rb') as input_md:
    model = pkl.load(input_md)

# Load the label encoder
with open('ec_vsfc.pkl', 'rb') as label_file:
    encoder = pkl.load(label_file)

# Define the emotion classes
class_list = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Streamlit app
st.header('Nhận diện cảm xúc từ câu phản hồi')
txt = st.text_area('Nhập câu bình luận của bạn:', '')

if st.button('Dự đoán'):
    if txt != '':
        # Transform the input text into a feature vector
        feature_vector = encoder.transform([txt])
        # Predict the emotion label
        label = int(model.predict(feature_vector)[0])
        st.header('Kết quả')
        st.text(f'Cảm xúc dự đoán: {class_list[label]}')
