import pandas as pd
import os


class Popularity:
    @staticmethod
    def make_recommend(dff, top_n=30, interval_days=7):
        max_data = dff['order_ts'].max()
        interval = pd.Timedelta(days=interval_days)
        month_item_counts = \
            dff.loc[(dff['order_ts'] >= (max_data - interval))
                    & (dff['order_ts'] <= max_data)].groupby(['item_id'])['user_id'].nunique().reset_index(
            ).rename(columns={'user_id': 'item_counts'})
        pop_items = month_item_counts.sort_values('item_counts', ascending=False).head(top_n)['item_id']
        return pop_items.values

    @staticmethod
    def make_candidates(dff,task, top_n=10, interval_days=7,
                        cand_path='', load=False, save_path='', save=False):
        if os.path.isfile(cand_path) and load:
            return pd.read_csv(cand_path)
        # Популярные товары по уникальным пользователям за interval_days от последней даты промежуток времени
        else:
            max_data = dff['order_ts'].max()
            interval = pd.Timedelta(days=interval_days)
            month_item_counts = \
                dff.loc[(dff['order_ts'] >= (max_data - interval))
                        & (dff['order_ts'] <= max_data)].groupby(['item_id'])['user_id'].nunique().reset_index(
                ).rename(columns={'user_id': 'item_counts'})
            pop_items = month_item_counts.sort_values('item_counts', ascending=False).head(top_n)[['item_id']]
            pop = dff.groupby('user_id').count().reset_index()[['user_id']]
            pop['item_id'] = pop['user_id'].apply(lambda x: pop_items.values)
            pop = pop.explode('item_id', ignore_index=True).astype(int)
            pop['rank'] = pop.groupby('user_id').cumcount() + 1
        if os.path.isfile(save_path) and save:
            pop.to_csv(save_path)
        return pop
