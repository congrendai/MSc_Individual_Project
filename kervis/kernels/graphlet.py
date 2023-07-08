import numpy as np
import networkx as nx
from kervis.kernels import Kernel
from itertools import combinations
from matplotlib import pyplot as plt

class Graphlet(Kernel):
    def __init__(self, k = 4, connected = True):
        self.k = k
        if connected:
            self.graphlets = [g for g in nx.graph_atlas_g() if len(g.nodes())==self.k and len(list(nx.connected_components(g)))==1]
        else:
            self.graphlets = [g for g in nx.graph_atlas_g() if len(g.nodes())==self.k]

    def get_feature(self, graph):
        feature = []
        C = combinations(list(graph), self.k)
        for c in C:
            for i in range(len(self.graphlets)):
                if nx.is_isomorphic(graph.subgraph(c), self.graphlets[i]):
                    feature.append((i))

        return feature

    def plot_all_graphlet(self, node_size = 10):
        graphlets_G = nx.Graph()
        for index, graphlet in enumerate(self.graphlets):  
            graphlets_G.add_nodes_from(np.array(graphlet.nodes())+index*self.k)
            graphlets_G.add_edges_from(np.array(graphlet.edges())+index*self.k)

        plt.figure(figsize=(10, 10), dpi=300)
        plt.margins(0.0)
        pos = nx.nx_agraph.pygraphviz_layout(graphlets_G)
        nx.draw(graphlets_G, pos=pos, node_color="tab:blue", width=0.5, node_size=node_size)

    def plot_graphlet(self, graphlet, node_size = 10):
        plt.figure(figsize=(10, 10), dpi=300)
        plt.margins(0.0)
        pos = nx.nx_agraph.pygraphviz_layout(graphlet)
        nx.draw(graphlet, pos=pos, node_color="tab:blue", width=0.5, node_size=node_size)