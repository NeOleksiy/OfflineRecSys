import streamlit as st
import pandas as pd
from recommender import recommend

st.set_page_config(
    page_title="Рекомендация",
    page_icon="👋",
)

# Настройка заголовка и текста
st.title("Рекомендер")


@st.cache_data
def load_candidates():
    candidates_data = pd.read_csv('data/datasets/final_range_candidates.csv')
    return candidates_data


# with st.spinner('Wait for it...'):
#     candidates = load_candidates()
# st.success('Done!')
candidates = load_candidates()


user_id = st.number_input('Введите user_id', value=7)
top_n = st.number_input('Введите кол-во рекомендуемых товаров', value=5)


if st.button('Рекомендовать'):
    with st.spinner('Wait for it...'):
        st.write(recommend(user_id, candidates, top_n))
    st.success('Done!')

