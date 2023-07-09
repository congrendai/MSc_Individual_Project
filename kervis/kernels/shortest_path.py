import networkx as nx
from kervis.kernels import Kernel

class ShortestPath(Kernel):
    def __init__(self, with_labels = True):
        self.with_labels = with_labels

    def get_feature(self, graph):
        if self.with_labels:
            pairs = []
            self.node_label_dict = dict(graph.nodes(data="label"))
            for path in nx.all_pairs_shortest_path_length(graph):
                # path[0]: target, path[1]: source, length: v
                pair = [(*sorted((path[0], k)), v) for k,v in path[1].items() if v != 0]
                pairs.extend(pair)

            # Remove duplicates
            pairs = list(set(pairs))

            # Assign node labels to node ids
            pairs_with_labels = [(self.node_label_dict[pair[0]], self.node_label_dict[pair[1]], pair[2]) for pair in pairs]

            return pairs_with_labels

