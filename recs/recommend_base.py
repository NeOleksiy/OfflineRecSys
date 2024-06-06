from abc import ABCMeta, abstractmethod


class BaseRecommender(metaclass=ABCMeta):
    @staticmethod
    def prepare_data(df):
        pass

    @abstractmethod
    def fit(self, coo_data):
        pass

