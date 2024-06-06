import unittest
import pandas as pd
from recommender import recommend
from trainer import clean_data
from recs.knn import KNN
import os


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_clear = pd.read_csv('data/datasets/wb_clean_data.csv')
        self.candidates = pd.read_csv('data/datasets/final_range_candidates.csv')

    def test_recommend(self):
        rec = recommend(1, self.candidates, 5)
        self.assertEqual(len(rec), 5)

    def test_cold_user(self):
        rec = recommend(99999999, self.candidates, 5)
        self.assertEqual(rec[1], 390)

    def test_Nan(self):
        self.assertEqual(self.candidates.isnull().values.all(), True)
        self.assertEqual(self.df_clear.isnull().values.all(), True)

    def test_coo_matrix_knn(self):
        coo_train = KNN.prepare_data(self.df_clear)
        self.assertEqual(coo_train.shape, (1057266, 1057266))

    def test_cleaner(self):
        clean_df = clean_data(self.df_clear)
        self.assertEqual(clean_df.duplicated().any(), False)

    def test_duplicates_in_candidates(self):
        self.assertEqual(self.candidates.duplicated(['user_id', 'item_id',
                                                     'rank_knn', 'rank_pop',
                                                     'rank_gru']).any(), False)

    def test_have_candidates(self):
        path = 'data/datasets/final_range_candidates.csv'
        self.assertEqual(os.path.isfile(path), True)


if __name__ == '__main__':
    unittest.main()
