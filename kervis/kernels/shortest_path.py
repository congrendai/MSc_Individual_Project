import networkx as nx
from kervis.kernels import Kernel

class ShortestPath(Kernel):
    def __init__(self, with_labels = True):
        self.with_labels = with_labels

    def get_feature(self, graph):
        if self.with_labels:
            pairs = []
            self.node_label_dict = dict(graph.nodes(data="label"))

            nodes = list(graph.nodes)

            shortest_paths = []
            for i in range(len(nodes)-1):
                for j in range(i+1, len(nodes)):
                    for path in nx.all_shortest_paths(graph, nodes[i], nodes[j]):
                        shortest_paths.append(path)


            # # Assign node labels to node ids
            path_with_labels = [(self.node_label_dict[shortest_path[0]], self.node_label_dict[shortest_path[-1]], len(shortest_path)-1) for shortest_path in shortest_paths]

            return path_with_labels

