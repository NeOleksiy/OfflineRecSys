import logging
import os
import sys
import pandas as pd
import torch
from clearml import Task, Dataset
from recs.gru4rec import GRU4Rec
from recs.knn import KNN
from recs.popularity import Popularity
from recs.range_catboost import RangeCatboost

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('Trainer RecSys')


def clean_data(df):
    df['order_ts'] = pd.to_datetime(df['order_ts']).dt.floor('s')
    df['item_n'] = df.groupby(['user_id', 'item_id', 'order_ts'])['user_id'].transform('count')
    df_clear = df.drop_duplicates()
    df_user = df_clear.groupby(by=['user_id'])[['item_n']].sum().reset_index().rename(
        columns={'item_n': 'item_purchased'})
    user_outliers = df_user[(df_user['item_purchased'] < 3) & (df_user['item_purchased'] > 100)]['user_id'].unique()
    df_clear = df_clear.drop(df_clear[df_clear['user_id'].isin(user_outliers)].index, axis=0)
    return df_clear


# Функция деления по квантилю времени для генерации позитивных и негативных примеров
def get_labels(data, q=0.7):
    data = data.sort_values(by='order_ts')
    date_threshold = data['order_ts'].quantile(q=q, interpolation='nearest')
    train_1 = data[(data['order_ts'] < date_threshold)]
    pred = data[(data['order_ts'] >= date_threshold)]
    pred = pred[pred['user_id'].isin(data['user_id'].unique())]
    return train_1, pred


# Обучение catboost и реранжирование кандидатов с последующим сохранением
def train_range_catboost(candidates_list, train_df, labels, models, save_path_cand, save_path_model, task):
    if os.path.isdir(save_path_cand):
        logger.error('Директория  для сохранения кандидатов не найдена')
        return
    if os.path.isdir(save_path_model):
        logger.error('Директория для сохранения модели не найдена')
        return
    range_ctb = RangeCatboost(task=task)
    logger.debug('Идёт препроцессинг для catboost')
    candidates = RangeCatboost.concat_candidates(candidates_list, models)
    user_df, item_df = RangeCatboost.generate_features(train_df)
    try:
        train_feat = RangeCatboost.dataset_for_catboost(candidates, labels, user_df, item_df, frac=0.09)
        train_feat = RangeCatboost.prepare_data(train_feat)
    except MemoryError:
        logger.error('Переполнение стека')
    logger.debug('Обучаем Catboost')
    range_ctb.fit(train_feat, save_path=save_path_model, save=True)
    logger.warning('Идёт переранжирование. Возможно переполнение памяти')
    try:
        cand = range_ctb.make_candidates(candidates, user_df, item_df,
                                         save_path=save_path_cand, save=True)
        logger.info('Переранжирование и сохранение прошло успешно')
        return cand
    except MemoryError:
        logger.error('Переполнение стека')


# Функция обучения knn и генерацией кандидатов с последующим сохранением
def train_knn(dff, save_path_cand, save_path_model, task, k=50):
    if os.path.isdir(save_path_cand):
        logger.error('Директория  для сохранения кандидатов не найдена')
        return
    if os.path.isdir(save_path_model):
        logger.error('Директория для сохранения модели не найдена')
        return
    knn = KNN(k=k, task=task)
    coo_train = KNN.prepare_data(dff)
    knn.fit(coo_train, save_path=save_path_model, save=True)
    return knn.make_candidates(dff, coo_train, save_path=save_path_cand, save=True)


# Функция перезаписывает Популярных кандидатов с последующим сохранением
def train_pop(dff, save_path_cand, task):
    if os.path.isdir(save_path_cand):
        logger.error('Директория  для сохранения кандидатов не найдена')
        return
    return Popularity.make_candidates(dff, save_path=save_path_cand, save=True, task=task)


# Функция обучения GRU4Rec и генерация кандидатов с последующим сохранением
def train_gru(dff, save_path_cand, save_path_model, task):
    if os.path.isdir(save_path_cand):
        logger.error('Директория  для сохранения кандидатов не найдена')
        return
    if os.path.isdir(save_path_model):
        logger.error('Директория для сохранения модели не найдена')
        return
    vocab = set(list(dff['item_id'].unique()) + ['<unk>', '<pad>'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        logger.warning('Обучение на cpu будет слишком долгим. Советуем перейти на gpu.')
    gru4rec = GRU4Rec(vocab, device=device, task=task)
    data, item2ind, ind2item = gru4rec.prepare_data(dff)
    _ = gru4rec.fit(data, num_epoch=5)
    user_ids = dff['user_id'].unique()
    logger.warning('Генерация кандидатов (долгая операция)')
    return gru4rec.make_candidates(dff, user_ids, item2ind, ind2item,
                                   save_path=save_path_cand, save=True)


if __name__ == "__main__":
    local_flag = False
    if len(sys.argv) == 1:
        logger.error('Для работы тренера нужно передать аргументы. Вот их список:\n'
                     '  "knn" - генерация, обучение и сохранение кандидатов knn\n'
                     '  "pop" - перезапись популярных кандидатов\n'
                     '  "gru" - генерация, обучение и сохранение кандидатов gru4rec\n'
                     '  "2lvl" - обучение, реранжирование и сохранение кандидатов модели второго уровня\n'
                     '  "rerange" - реранжирование и сохранение кандидатов модели второго уровня')
        sys.exit()

    logger.debug('Читаем csv файл с сырыми данными')
    if 'local' in sys.argv:
        local_flag = True
        df = pd.read_csv('data/datasets/wb_school_task.csv')
    else:
        dataset = Dataset.get(dataset_id='40368742aea745118b0522f05c74c86a')
        data_path = Dataset.get(dataset_id='40368742aea745118b0522f05c74c86a',
                                alias='datasets').get_local_copy()
        df = pd.read_csv(f'{data_path}/datasets/wb_school_task.csv')
    logger.debug('Удаляем дубликаты по user_id, item_id, order_ts')
    df['order_ts'] = pd.to_datetime(df['order_ts']).dt.floor('s')
    df['item_n'] = df.groupby(['user_id', 'item_id', 'order_ts'])['user_id'].transform('count')
    df_clear = df.drop_duplicates()
    df_user = df_clear.groupby(by=['user_id'])[['item_n']].sum().reset_index().rename(
        columns={'item_n': 'item_purchased'})
    user_outliers = df_user[(df_user['item_purchased'] < 3) & (df_user['item_purchased'] > 100)]['user_id'].unique()
    df_clear = df_clear.drop(df_clear[df_clear['user_id'].isin(user_outliers)].index, axis=0)
    train, label = get_labels(df_clear, q=0.7)

    if 'knn' in sys.argv:
        task_knn = Task.init(project_name='wb-rec-sys', task_name='fit_knn')
        logger.debug('Обучаем KNN')
        train_knn(train, save_path_cand='data/datasets/candidates_knn.csv',
                  save_path_model='/models', task=task_knn)
        task_knn.close()
        logger.info('Обучение KNN завершено')
    if 'pop' in sys.argv:
        task_pop = Task.init(project_name='wb-rec-sys', task_name='fit_pop')
        logger.debug('Перезаписываем Popularity')
        train_pop(train, save_path_cand='data/datasets/pop_candidates.csv', task=task_pop)
        task_pop.close()
        logger.info('Перезапись Popularity завершена')
    if 'gru' in sys.argv:
        task_gru = Task.init(project_name='wb-rec-sys', task_name='fit_gru')
        logger.info('Обучаем GRU')
        train_gru(train, save_path_cand='data/datasets/candidates_gru.csv',
                  save_path_model='/models/gru_model.pt', task=task_gru)
        task_gru.close()
        logger.info('Обучение GRU4Rec завершено')
    if '2lvl' in sys.argv:
        task_ctb = Task.init(project_name='wb-rec-sys', task_name='fit_catboost')
        logger.debug('Читаем кандидатов с моделей:')
        cols = ['user_id', 'item_id', 'rank']
        model_list = ['knn', 'pop', 'gru']
        try:
            knn_cand = pd.read_csv('data/datasets/candidates_knn.csv')[cols]
        except FileNotFoundError:
            logger.error('Файл knn кандидатов не найден')
        try:
            pop_cand = pd.read_csv('data/datasets/pop_candidates.csv')[cols]
        except FileNotFoundError:
            logger.error('Файл popularity кандидатов не найден')
        try:
            gru_cand = pd.read_csv('data/datasets/candidates_gru.csv')[cols]
        except FileNotFoundError:
            logger.error('Файл gru кандидатов не найден')
        cand_list = [knn_cand, pop_cand, gru_cand]
        logger.debug('Начало обучения ранжирующей модели catboost')
        train_range_catboost(cand_list, train, label, model_list,
                             save_path_cand='data/datasets/final_range_candidates.csv',
                             save_path_model='models/ctb_model', task=task_ctb)
        task_ctb.close()
        logger.info('Обучение и переранжирование завершено')

    if 'rerange' in sys.argv:
        task_re_ctb = Task.init(project_name='wb-rec-sys', task_name='rerange_catboost')
        logger.debug('Читаем кандидатов с моделей:')
        cols = ['user_id', 'item_id', 'rank']
        model_list = ['knn', 'pop', 'gru']
        try:
            knn_cand = pd.read_csv('data/datasets/candidates_knn.csv')[cols]
        except FileNotFoundError:
            logger.error('Файл knn кандидатов не найден')
        try:
            pop_cand = pd.read_csv('data/datasets/pop_candidates.csv')[cols]
        except FileNotFoundError:
            logger.error('Файл popularity кандидатов не найден')
        try:
            gru_cand = pd.read_csv('data/datasets/candidates_gru.csv')[cols]
        except FileNotFoundError:
            logger.error('Файл gru кандидатов не найден')
        cand_list = [knn_cand, pop_cand, gru_cand]
        range_ctb = RangeCatboost(model_path='models/ctb_model', task=task_re_ctb)
        user_df, item_df = RangeCatboost.generate_features(train)
        candidates = RangeCatboost.concat_candidates(cand_list, model_list)
        logger.warning('Идёт переранжирование. Возможно переполнение памяти')
        try:
            candidates = range_ctb.make_candidates(candidates, user_df, item_df,
                                                   save_path='data/datasets/final_range_candidates.csv', save=True)
        except MemoryError:
            logger.error('Переполнение стека')
        task_re_ctb.close()
        logger.info('Переранжирование кандидатов завершено')
    if not local_flag:
        logger.info('Сохраняем изменения в ClearML')
        dataset = Dataset.create(dataset_name='candidates_datasets',
                                 dataset_project='data/datasets',
                                 parent_datasets=['40368742aea745118b0522f05c74c86a'])

        dataset.finalize(auto_upload=True)
