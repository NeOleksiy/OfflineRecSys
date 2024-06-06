import sys
import logging
import time
from recs.popularity import Popularity

import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('Trainer RecSys')


def recommend(user_id, candidates, top_n=5):
    if user_id in candidates['user_id'].values:
        return candidates[candidates['user_id'] == user_id].head(top_n)['item_id'].values
    else:
        logger.debug('Пользователь холодный, делаю неперсональные рекомендации')
        logger.debug('Читаем файл с данными')
        try:
            df = pd.read_csv('data/datasets/wb_school_task.csv')
        except FileNotFoundError:
            logger.error('Файл не найден')
        df['order_ts'] = pd.to_datetime(df['order_ts']).dt.floor('s')
        df['item_n'] = df.groupby(['user_id', 'item_id', 'order_ts'])['user_id'].transform('count')
        df_clear = df.drop_duplicates()
        logger.info('Файл прочитан')
        return Popularity.make_recommend(df_clear, top_n=5, interval_days=7)


if __name__ == "__main__":
    try:
        user_id = input("Введите user_id: ")
    except EOFError:
        logger.error("Ошибка: не был введен user_id.")
    logger.debug('Читаем csv файл с кандидатами')
    candidates = pd.read_csv('data/datasets/final_range_candidates.csv')
    logger.info('Файл прочитан')
    print(recommend(int(user_id), candidates, top_n=5))
