"""UI using streamlit and a LogisticRegr model"""
import streamlit as st
from src.dataprocessing.text_cleaner import clean_text
from src.models.train_for_app import train_model_for_app

st.title('Класифициране на ревюта на хотел')

user_input = st.text_area("Въведете вашето ревю:")
model, vectorizer = train_model_for_app()

if st.button('Класифицирай'):
    CLEAN_INPUT = clean_text(user_input)
    vectorized_input = vectorizer.transform([CLEAN_INPUT])
    prediction = model.predict(vectorized_input)

    st.write(f"Предсказан рейтинг: {prediction[0]}")
