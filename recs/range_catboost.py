import os
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from recs.recommend_base import BaseRecommender

warnings.filterwarnings('ignore')


class RangeCatboost(BaseRecommender):
    def __init__(self, task, model_path=''):
        self.task = task
        if os.path.isfile(model_path):
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
        else:
            self.model = CatBoostClassifier()

    @staticmethod
    def prepare_data(df):
        drop_col = ['user_id', 'item_id']
        target_col = ['target']
        cat_col = ['day_last_order_y', 'day_last_order_x', 'month_active', 'hour_active']
        bad_columns = ['order_median', 'user_n', 'orders_count_8march', 'day_last_order_y',
                       'month_item_counts', 'week_item_counts',
                       'rank_knn', 'rank_pop', 'rank_gru']
        train_feat = df.replace([np.inf, -np.inf], np.nan)
        train_feat[bad_columns] = train_feat[bad_columns].astype(int, errors='ignore')
        train_feat[cat_col] = train_feat[cat_col].astype(str, errors='ignore')
        X_train, y_train = train_feat.drop(drop_col + target_col, axis=1), train_feat[target_col]
        X_train = X_train.fillna(X_train.mode().iloc[0])
        train = Pool(data=X_train, label=y_train, cat_features=cat_col)
        return train

    @staticmethod
    def dataset_for_catboost(candidates, label, user_df, item_df, frac=0.07):

        pos = candidates.merge(label,
                               on=['user_id', 'item_id'],
                               how='inner')

        pos['target'] = 1
        neg = candidates.set_index(['user_id', 'item_id']) \
            .join(label.set_index(['user_id', 'item_id']))

        neg = neg[neg['order_ts'].isnull()].reset_index()
        # Сэмплим лишь часть негативных кандидатов, чтобы держать баланс классов
        neg = neg.sample(frac=frac)
        neg['target'] = 0
        select_col = ['user_id', 'item_id', 'rank_knn', 'rank_pop', 'rank_gru', 'target']
        users = label['user_id'].unique()
        ctb_train = shuffle(
            pd.concat([
                pos[pos['user_id'].isin(users)],
                neg[neg['user_id'].isin(users)]
            ])[select_col]
        )
        train_feat = ctb_train.merge(user_df, on=['user_id'], how='left').merge(
            item_df, on=['item_id'], how='left')
        return train_feat

    @staticmethod
    def concat_candidates(candidates_list, model_list):
        cols = ['user_id', 'item_id']
        candidates = pd.concat(candidates_list, axis=0, join='inner')[cols]
        for name, cand in zip(model_list, candidates_list):
            candidates = candidates.merge(cand.rename(columns={'rank': f'rank_{name}'}),
                                          on=['user_id', 'item_id'], how='left')
        candidates = candidates.fillna(100)
        candidates = candidates.drop_duplicates(['user_id', 'item_id', 'rank_knn', 'rank_pop', 'rank_gru'])
        return candidates

    @staticmethod
    def generate_features(df):
        df_copy = df.copy()
        df_time = df.set_index('order_ts')
        user_df = df.groupby(by=['user_id'])[['item_n']].sum().reset_index(

        ).rename(columns={'item_n': 'item_purchased'})
        # Час в дне наибольшей активности
        df_time_h = df_copy.copy()
        df_time_h['hour_active'] = df_time_h['order_ts'].dt.hour
        df_hour_active = df_time_h.groupby(['user_id', 'hour_active']).size().reset_index(name='count')
        df_hour_active = df_hour_active.loc[df_hour_active.groupby('user_id')['count'].idxmax()][
            ['user_id', 'hour_active']]
        user_df = user_df.merge(df_hour_active, on='user_id')
        # Месяц наибольшей активности
        df_time_m = df_copy.copy()
        df_time_m['month_active'] = df_time_m['order_ts'].dt.month
        df_month_active = df_time_m.groupby(['user_id', 'month_active']).size().reset_index(name='count')
        df_month_active = df_month_active.loc[df_month_active.groupby('user_id')['count'].idxmax()][
            ['user_id', 'month_active']]
        user_df = user_df.merge(df_month_active, on='user_id')
        # Средний интервал между покупками
        df_time_i = df_copy.copy()
        df_time_i.sort_values(by=['user_id', 'order_ts'], inplace=True)
        df_time_i['interval'] = df_time_i.groupby('user_id')['order_ts'].diff().dt.days.astype(float)
        result = df_time_i.groupby(['user_id'])['interval'].mean().reset_index()
        df_interval = result[['user_id', 'interval']]
        user_df = user_df.merge(df_interval, on='user_id')
        # Число заказов пользователя
        df_order_n = df_copy.groupby(['user_id', 'order_ts'])[['item_n']].count().reset_index().groupby(['user_id'])[
            ['item_n']].count().reset_index(
        ).rename(columns={'item_n': 'order_n'})
        user_df = user_df.merge(df_order_n, on='user_id')
        # Медиана,среднее и максимальное число товаров в заказе
        dff = df_copy
        df_order_median = dff.groupby(['user_id', 'order_ts']).count().reset_index().groupby(['user_id'])[
            ['item_id']].median().reset_index(
        ).rename(columns={'item_id': 'order_median'})
        user_df = user_df.merge(df_order_median, on='user_id')
        df_order_max = dff.groupby(['user_id', 'order_ts']).count().reset_index().groupby(['user_id'])[
            ['item_id']].max().reset_index(
        ).rename(columns={'item_id': 'order_max'})
        user_df = user_df.merge(df_order_max, on='user_id')
        df_order_mean = dff.groupby(['user_id', 'order_ts']).count().reset_index().groupby(['user_id'])[
            ['item_id']].mean().reset_index(
        ).rename(columns={'item_id': 'order_mean'})
        user_df = user_df.merge(df_order_mean, on='user_id')
        # Активность, сколько дней не делал заказов считая с последней известно даты в датасете
        last_order_date = df_copy['order_ts'].max()
        last_order_user = df_copy.groupby('user_id')[['order_ts']].max().reset_index()
        last_order = last_order_user['order_ts'].apply(lambda x: np.abs(x - last_order_date).days)
        last_order_user['order_ts'] = last_order
        last_order_user = last_order_user.rename(columns={'order_ts': 'day_last_order'})
        day_last_order = last_order_user[['user_id', 'day_last_order']]
        user_df = user_df.merge(day_last_order, on='user_id')
        # Число заказов у товара
        df_time_f = df_time.copy()
        df_item = df_time_f.groupby(by=['item_id'])[['item_n']].sum().rename({'item_n': 'user_n'})
        item_df = df_item.reset_index().rename(columns={'item_n': 'user_n'})
        # Cредний интервал между заказами для каждого товара по дням
        dff = df_copy.sort_values(by=['user_id', 'order_ts'])
        dff['time_diff'] = dff.groupby(['item_id', 'user_id'])['order_ts'].diff().dt.days.astype(float)
        average_interval_per_item = dff.groupby('item_id')[['time_diff']].mean().reset_index()
        item_df = item_df.merge(average_interval_per_item, on='item_id')
        # Кол-во заказов в интервале 3-8 марта
        filtered_data = df_copy[(df_copy['order_ts'].dt.date >= pd.Timestamp('2023-03-3').date()) & (
                df_copy['order_ts'].dt.date <= pd.Timestamp('2023-03-8').date())]
        march8 = filtered_data.groupby(['item_id']).size().reset_index(name='orders_count_8march')
        item_df = item_df.merge(march8, on='item_id')
        # Сколько дней товар не покупали
        dff = df_copy
        last_order_date = dff['order_ts'].max()
        last_order_item = dff.groupby('item_id')[['order_ts']].max().reset_index()
        last_order = last_order_item['order_ts'].apply(lambda x: np.abs(x - last_order_date).days)
        last_order_item['order_ts'] = last_order
        last_order_item = last_order_item.rename(columns={'order_ts': 'day_last_order'})
        day_not_buy = last_order_item[['item_id', 'day_last_order']]
        item_df = item_df.merge(day_not_buy, on='item_id')
        # Кол-во заказов у товара за последние 7 дней
        dff = df_copy
        week_item_counts = \
            dff.loc[(dff['order_ts'] >= '2023-03-24') & (dff['order_ts'] <= '2023-04-01')].groupby(['item_id'])[
                'item_n'].sum().reset_index(
            ).rename(columns={'item_n': 'week_item_counts'})
        item_df = item_df.merge(week_item_counts, on='item_id', how='left').fillna(0)
        # Кол-во заказов у товара за последний месяц
        month_item_counts = \
            dff.loc[(dff['order_ts'] >= '2023-03-01') & (dff['order_ts'] <= '2023-04-01')].groupby(['item_id'])[
                'item_n'].sum().reset_index(
            ).rename(columns={'item_n': 'month_item_counts'})
        item_df = item_df.merge(month_item_counts, on='item_id', how='left').fillna(0)
        # Время расписаная по фичам
        return user_df, item_df

    def fit(self, train, save_path='', save=False):
        self.model.fit(train,
                       early_stopping_rounds=100,
                       verbose=100)
        self.task.add_parameters(self.model.get_all_params())
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(train.columns)[sorted_idx])
        plt.title('Feature Importance')
        self.task.get_logger().report_plot(title='Feature Importance',
                                           iteration=0, xlabel='X', ylabel='Y', plot=plt)
        if os.path.isdir(save_path) and save:
            self.task.upload_artifact('ctb_model.pth', artifact_object=self.model)
            self.model.save_model('models/ctb_model')

    def make_candidates(self, candidates, user_df, item_df, cand_path='', load=False, save_path='', save=False):
        if os.path.isfile(cand_path) and load:
            return pd.read_csv(cand_path)
        else:
            cat_col = ['day_last_order_y', 'day_last_order_x', 'month_active', 'hour_active']
            drop_col = ['user_id', 'item_id']
            score_feat = candidates.merge(user_df, on=['user_id'], how='left')
            score_feat = score_feat.merge(item_df, on=['item_id'], how='left')
            score_feat = score_feat.replace([np.inf, -np.inf], np.nan)
            score_feat = score_feat.fillna(score_feat.mode().iloc[0])
            score_feat[cat_col] = score_feat[cat_col].astype(str, errors='ignore')
            score_feat = score_feat.drop(drop_col, axis=1, errors='ignore')
            ctb_prediction = self.model.predict_proba(score_feat)
            candidates['ctb_pred'] = ctb_prediction[:, 1]
            candidates = candidates.sort_values(
                by=['user_id', 'ctb_pred'], ascending=[True, False])
            candidates['rank_ctb'] = candidates.groupby('user_id').cumcount() + 1
            if os.path.isfile(save_path) and save:
                candidates.to_csv(save_path)
        return candidates
