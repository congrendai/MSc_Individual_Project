import numpy as np
import networkx as nx
from kervis.kernels import Kernel
from itertools import combinations
from matplotlib import pyplot as plt

class Graphlet(Kernel):
    """
    This class is for graphlet kernel

    Parameters
    ----------
    k: int
        the size of the graphlets

    connected: bool (default: True)
        whether to use connected graphlets only

    Attributes
    ----------
    X: numpy.ndarray
        the feature matrix

    attributes: list
        a list of the attributes of the feature matrix
        
    graphlets: list
        a list of the graphlets
    """
    def __init__(self, k = 3, connected = True):
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

    def plot_all_graphlets(self, node_size = 5):
        """
        This function is used to plot all graphlets

        Parameters
        ----------
        node_size: int
            the size of the nodes in the plot

        Returns
        -------
        None
        """
        graphlets_G = nx.Graph()
        for index, graphlet in enumerate(self.graphlets):  
            graphlets_G.add_nodes_from(np.array(graphlet.nodes())+index*self.k)
            graphlets_G.add_edges_from(np.array(graphlet.edges())+index*self.k)

        plt.figure(figsize=(10, 10), dpi=300)
        pos = nx.nx_agraph.pygraphviz_layout(graphlets_G)
        nx.draw(graphlets_G, pos=pos, node_color="tab:blue", width=0.5, node_size=node_size)

    def plot_graphlet(self, graphlet_index, node_size = 80):
        """
        This function is used to plot a graphlet

        Parameters
        ----------
        graphlet_index: int
            the index of the graphlet to plot

        node_size: int
            the size of the nodes in the plot

        Returns
        -------
        None
        """
        graphlet = self.graphlets[graphlet_index]
        plt.figure(figsize=(10, 10), dpi=300)
        pos = nx.nx_agraph.pygraphviz_layout(graphlet)
        nx.draw(graphlet, pos=pos, node_color="tab:blue", width=1, node_size=node_size)