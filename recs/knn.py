from recs.recommend_base import BaseRecommender
import numpy as np
import pandas as pd
import os
from scipy.sparse import coo_matrix
from implicit.nearest_neighbours import CosineRecommender
from clearml import Dataset, Task, OutputModel


class KNN(BaseRecommender):
    def __init__(self, task, k=30, path='', load=False):
        self.task = task
        self.task.add_parameters({'k': k})
        if os.path.isfile(path) and load:
            self.model = CosineRecommender(K=k)
            self.model.load(path)
        else:
            self.model = CosineRecommender(K=k)

    @staticmethod
    def prepare_data(df):
        ALL_USERS = df['user_id'].unique().tolist()
        ALL_ITEMS = df['item_id'].unique().tolist()
        user_ids = dict(list(enumerate(ALL_USERS)))
        item_ids = dict(list(enumerate(ALL_ITEMS)))
        row = df['user_id'].values
        col = df['item_id'].values
        data = np.ones(df.shape[0])
        n = max(max(row), max(col)) + 1
        coo_train = coo_matrix((data, (row, col)), shape=(n, n))
        return coo_train

    def fit(self, coo_data, save_path='', save=False):
        csr_data = coo_data.tocsr()
        self.model.fit(csr_data)
        if os.path.isdir(save_path) and save:
            self.task.upload_artifact('knn_model.pth', artifact_object=self.model)
            self.model.save(save_path + '/KNN_model')

    def make_candidates(self, train, coo_train, n=30, cand_path='', load=False, save_path='', save=False):
        if os.path.isfile(cand_path) and load:
            return pd.read_csv(cand_path)
        else:
            user_ids = np.array(train['user_id'].unique())
            ids, scores = self.model.recommend(user_ids,
                                               coo_train.tocsr()[user_ids],
                                               N=n,
                                               filter_already_liked_items=True)
            preds = []
            for i, userid in enumerate(user_ids):
                preds.append((userid, ids[i], scores[i]))
            candidates = pd.DataFrame(preds, columns=['user_id', 'item_id', 'rank'])
            candidates = candidates.explode(['item_id', 'rank'])
            candidates['rank'] = candidates.groupby('user_id').cumcount() + 1
            if os.path.isfile(save_path) and save:
                candidates.to_csv(save_path)
        return candidates
