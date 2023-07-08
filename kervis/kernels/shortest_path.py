import numpy as np
from kervis.kernels import Kernel
import networkx as nx
from collections import Counter
from multiprocessing import Pool

class ShortestPath(Kernel):
    def __init__(self, with_labels = True):
        self.with_labels = with_labels

    def get_feature(self, graph):
        if self.with_labels:
            pairs = []
            node_label_dict = dict(graph.nodes(data="label"))
            for length in nx.all_pairs_shortest_path_length(graph):
                pair = [(*sorted((length[0], k)), v) for k,v in length[1].items() if v != 0]
                pairs.extend(pair)

            # Remove duplicates
            pairs = list(set(pairs))

            # Assign node labels to node ids
            pairs_with_labels = [(node_label_dict[pair[0]], node_label_dict[pair[1]], pair[2]) for pair in pairs]

            return pairs_with_labels