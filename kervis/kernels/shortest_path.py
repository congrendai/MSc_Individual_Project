import networkx as nx
from collections import Counter

class ShortestPath():
    def __init__(self, with_labels = True):
        self.with_labels = with_labels
            

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