import numpy as np
import networkx as nx
from kervis.kernels import Kernel
from itertools import combinations
from matplotlib import pyplot as plt

class Graphlet(Kernel):
    """
    Graphlet kernel

    Parameters
    ----------
    k : int, default=3
        The size of graphlet

    connected : bool, default=False
        If True, only connected graphlets are considered

    Attributes
    ----------
    name : str
        The name of the kernel

    graphlets : list of networkx.Graph
        The list of graphlets
    """

    def __init__(self, k = 3, connected = False):
        self.k = k
        self.name = "GL_" + str(self.k)
        if connected:
            self.graphlets = [g for g in nx.graph_atlas_g() if len(g.nodes())==self.k and len(list(nx.connected_components(g)))==1]
        else:
            self.graphlets = [g for g in nx.graph_atlas_g() if len(g.nodes())==self.k]

    def get_feature(self, graph):
        """
        Get the feature vector of a graph

        Parameters
        ----------
        graph : networkx.Graph
            The graph

        Returns
        -------
        feature : list of int
            The feature vector
        """
        feature = []
        C = combinations(list(graph), self.k)
        for c in C:
            for i in range(len(self.graphlets)):
                if nx.is_isomorphic(graph.subgraph(c), self.graphlets[i]):
                    feature.append((i))

        return feature

    def plot_all_graphlet(self, node_size = 5):
        graphlets_G = nx.Graph()
        for index, graphlet in enumerate(self.graphlets):  
            graphlets_G.add_nodes_from(np.array(graphlet.nodes())+index*self.k)
            graphlets_G.add_edges_from(np.array(graphlet.edges())+index*self.k)

        plt.figure(figsize=(10, 10), dpi=100)
        pos = nx.nx_agraph.pygraphviz_layout(graphlets_G)
        nx.draw(graphlets_G, pos=pos, node_color="tab:blue", width=0.5, node_size=node_size)

    def plot_graphlet(self, graphlet_index, node_size = 80):
        graphlet = self.graphlets[graphlet_index]
        plt.figure(figsize=(10, 10), dpi=300)
        plt.margins(0.0)
        pos = nx.nx_agraph.pygraphviz_layout(graphlet)
        nx.draw(graphlet, pos=pos, node_color="tab:blue", width=1, node_size=node_size)