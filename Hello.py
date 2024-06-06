import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Началная страница UI👋")
st.markdown(
    """
    Небольшой UI в котором можно будет пощупать рекомендательную систему,посмотреть на данные и посмотреть на метрики.

"""
)

st.image('data/смешной кот.jpg', width=400)
