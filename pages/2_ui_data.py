import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.request import urlopen
import json

st.set_page_config(
    page_title="Просмотр данных",
    page_icon="👋",
)

# Настройка заголовка и текста
st.title("Данные по рекомендательной системе")

# Настройка боковой панели
st.sidebar.title("Выберите опцию")


DATA = 'data/datasets/wb_clean_data.csv'
CANDIDATES = 'data/datasets/final_range_candidates.csv'
METRICS = 'data/metrics_2lvl.csv'
DATE_COLUMN = 'order_ts'


# Создадим функцию для загрузки данных
@st.cache_data
def load_data():
    cols = ['user_id', 'item_id', 'order_ts', 'item_n']
    df = pd.read_csv(DATA, parse_dates=[DATE_COLUMN])[cols]
    return df


def load_metrics():
    metrics = pd.read_csv(METRICS).rename(columns={'Unnamed: 0': 'k-recs'})
    return metrics


@st.cache_data
def popular_items(df, n, low_date, high_data):
    month_item_counts = df.loc[(df['order_ts'] >= low_date)
                               & (df['order_ts'] <= high_data)].groupby(['item_id']
                                                                        )['user_id'].nunique().reset_index(
    ).rename(columns={'user_id': 'item_counts'})
    pop_items = month_item_counts.sort_values('item_counts', ascending=False).head(n)
    return pop_items[['item_id', 'item_counts']]


# Применим функцию
df = load_data()
metrics = load_metrics()
max_data = df['order_ts'].max()
min_data = df['order_ts'].min()
interval = pd.Timedelta(days=14)
df_part = df.loc[(df['order_ts'] >= (max_data - interval)) & (df['order_ts'] <= max_data)]




select_event = st.sidebar.selectbox('----', ('показать датасет', 'популярные товары', 'метрики'))
if select_event == 'показать датасет':
    st.subheader('Очищенные данные')
    st.markdown(
        "#### Таблица с взаимодействия пользователя и товар, очищенные от выбросов и дубликатов за 14 последних дней.\n"
        "Под дубликатами подразумевались взаимодействия, совершённые в одной и тоже время.\n"
        "Их также можно расценивать как кол-во купленных за раз товаров, за что и отвечает столбец item_n.\n"
        "Под выбросами же подрузамевались пользователи, сделавшие меньше 3х покупок и больше 100.")
    st.write(df_part)

if select_event == 'популярные товары':
    st.subheader('Популярные товары')
    st.markdown(
        "#### Таблица популярных товаров за промежуток времени.\n")
    min_selection, max_selection = st.sidebar.slider("Временной промежуток", min_value=min_data.to_pydatetime(),
                                                     max_value=max_data.to_pydatetime(),
                                                     value=[min_data.to_pydatetime(), max_data.to_pydatetime()])
    top_n = st.sidebar.slider("Кол-во", min_value=1, max_value=100,
                              value=10)
    st.write(popular_items(df, top_n, min_selection, max_selection))

if select_event == 'метрики':
    st.subheader('Метрики на отложенной тестовой выборке')
    st.markdown(
        "#### Метрики измерены на последних 10ти дней по среднему значению по пользователям.\n"
        "#### Какие метрики представлены:\n"
        "######   hit_rate - аналог accuracy, проще говоря среднее от суммы 1 - купил, 0 - не купил.\n"
        "######   precision - показывает долю релевантных товаров среди рекомендованных.\n"
        "######   recall - похожа на precision, метрика отвечает за кол-во товаров, релевантных пользователю.\n"
        "######   map - метрика ранжирования, среднее средней точности отранжированных объектов.\n"
        "######   ndcg - метрика ранжирования, которая учитывает не только релевантность, но и порядок.\n"
        "######   coverage - покрытие, доля рекомендуемых товаров от общего числа.")
    st.write(metrics)
