import numpy as np
import networkx as nx
from collections import Counter
from multiprocessing import Pool

class Kernel():
    def __init__(self):
        pass

    def get_feature(self):
        pass

    def fit_transform(self, G):
        with Pool() as pool:
            results = pool.map(self.get_feature, G)
            features = [result for result in results]

        feature_counters = [Counter(feature) for feature in features]

        # get all the keys of all graphs (duplicated keys included)
        feature_counter_keys = [key for feature_counter in feature_counters for key in feature_counter.keys()]
        self.attributes = list(set(feature_counter_keys))

        sparse_matrix = np.zeros([len(feature_counters), len(self.attributes)])
        for i, feature_counter in enumerate(feature_counters):
            for key, value in feature_counter.items():
                for j, attribute in enumerate(self.attributes):
                    if key == attribute:
                        sparse_matrix[i,j] = value

        self.X = sparse_matrix