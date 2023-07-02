import numpy as np
import networkx as nx
from multiprocessing import Pool
from collections import Counter
from itertools import combinations
from matplotlib import pyplot as plt

class Graphlet():
    def __init__(self, k = 4):
        self.k = k
        self.graphlets = [g for g in nx.graph_atlas_g() if len(g.nodes())==self.k and len(list(nx.connected_components(g)))==1] # and len(list(nx.connected_components(g)))==1

    def get_feature(self, graph):
        feature = []
        C = combinations(list(graph), self.k)
        for c in C:
            for i in range(len(self.graphlets)):
                if nx.is_isomorphic(graph.subgraph(c), self.graphlets[i]):
                    feature.append((c, i))

        return feature

    def fit_transform(self, G):
        with Pool() as pool:
            results = pool.map(self.get_feature, G)

        features = [result for result in results]

        return features


    def plot_graphlet(self):
        graphlets_G = nx.Graph()
        for index, graphlet in enumerate(self.graphlets):  
            graphlets_G.add_nodes_from(np.array(graphlet.nodes())+index*self.k)
            graphlets_G.add_edges_from(np.array(graphlet.edges())+index*self.k)

        plt.figure(figsize=(10, 10), dpi=300)
        plt.margins(0.0)
        pos = nx.nx_agraph.pygraphviz_layout(graphlets_G)
        nx.draw(graphlets_G, pos=pos, node_color="tab:blue", width=0.5, node_size=1)