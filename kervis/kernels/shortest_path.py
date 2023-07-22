import networkx as nx
from kervis.kernels import Kernel

class ShortestPath(Kernel):
    """
    This class is for the shortest path kernel

    Parameters
    ----------
    with_labels: bool
        whether to use node labels or not

    Attributes
    ----------
    X: numpy.ndarray
        the feature matrix

    attributes: list
        a list of the attributes of the feature matrix

    node_label_dict: dict
        a color map dictionary of the node labels
    """
    def __init__(self, with_labels = True):
        self.with_labels = with_labels

    def get_feature(self, graph):
        """
        This function is used to get the shortest paths of a graph
        
        Parameters
        ----------
        graph: networkx.classes.graph.Graph
            a graph of the dataset
            
        Returns
        -------
        path_with_labels: list
            a list of (a,b,c) where a is the label of the first node, b is the label of the last node, and c is the length of the shortest path
        """
        if self.with_labels:
            self.node_label_dict = dict(graph.nodes(data="label"))

            nodes = list(graph.nodes)

            shortest_paths = []
            for i in range(len(nodes)-1):
                for j in range(i+1, len(nodes)):
                    try:
                        # some nodes are isolated from the graph
                        for path in nx.all_shortest_paths(graph, nodes[i], nodes[j]):
                            shortest_paths.append(path)
                    except:
                        pass


            # # Assign node labels to node ids
            path_with_labels = [(self.node_label_dict[shortest_path[0]], self.node_label_dict[shortest_path[-1]], len(shortest_path)-1) for shortest_path in shortest_paths]

            return path_with_labels

