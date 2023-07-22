import numpy as np
from collections import Counter
from multiprocessing import Pool

class Kernel():
    """
    This is the base class for all kernels

    Parameters
    ----------
    None

    Attributes
    ----------
        X: numpy.ndarray
            the feature matrix
        attributes: list
            a list of the attributes of the feature matrix
    """
    def __init__(self):
        # this function is used to initialize the kernel
        pass

    def get_feature(self, graph):
        """
        this function is used to get the feature of a graph

        Parameters
        ----------
        graph: networkx.classes.graph.Graph
            a graph of the dataset

        Returns
        -------
        feature: list
            a list of the features of the graph
        """ 
        pass

    def fit_transform(self, G):
        """
        this function is used to get the feature matrix of the dataset

        Parameters
        ----------
        G: list
            a list of the graphs of the dataset

        Returns
        -------
        None
        """
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