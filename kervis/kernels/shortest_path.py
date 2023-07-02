import networkx as nx
from multiprocessing import Pool
from collections import Counter

class ShortestPath():
    def __init__(self, with_labels = True):
        self.with_labels = with_labels

    def get_feature(self, graph):
        lengths = []
        node_label_dict = dict(graph.nodes(data="label"))
        for length in nx.all_pairs_shortest_path_length(graph):
            pair = [(*sorted((length[0], k)), v) for k,v in length[1].items() if v != 0]
            lengths.extend(pair)
        
        # Remove duplicates
        lengths = list(set(lengths))

        # Assign node labels to node ids
        if self.with_labels:
            lengths_with_labels = []
            for length in lengths:
                lengths_with_labels.append((node_label_dict[length[0]], node_label_dict[length[1]], length[2]))

            return lengths_with_labels
        else:
            return lengths
            

    def fit_transform(self, G):
        if self.with_labels:
            lengths = []
            lengths_with_labels = []
            for g in G:
                pairs = []
                pairs_with_labels = []
                node_label_dict = dict(g.nodes(data="label"))
                for length in nx.all_pairs_shortest_path_length(g):
                    pair = [(*sorted((length[0], k)), v) for k,v in length[1].items() if v != 0]
                    pairs.extend(pair)
                
                # Remove duplicates
                pairs = list(set(pairs))
                lengths.append(pairs)

                # Assign node labels to node ids
                for pair in pairs:
                    pairs_with_labels.append((node_label_dict[pair[0]], node_label_dict[pair[1]], pair[2]))

                lengths_with_labels.append(pairs_with_labels)

            return lengths, lengths_with_labels