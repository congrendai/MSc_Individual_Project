import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
from kervis.utils.utils import read_data
from kervis.utils.utils import fetch_dataset

class Dataset():
    """
    This class is for augmenting the fetch_dataset function in kervis.utils.utils

    Parameters
    ----------
    name: str
        the name of the dataset

    cmap: str (default: "coolwarm")

    from_TUDataset: bool (default: True)
        whether to fetch the dataset from TUDataset

    verbose: bool (default: False)
        whether to print the information of the dataset

    data_home: str (default: None)
        the path to the directory containing the datasets

    download_if_missing: bool (default: True)
        whether to download the dataset if it is not found

    with_classes: bool (default: True)
        whether to fetch the dataset with classes

    produce_labels_nodes: bool (default: False)
        whether to produce labels for nodes

    prefer_attr_nodes: bool (default: False)
        whether to prefer node attributes

    prefer_attr_edges: bool (default: False)
        whether to prefer edge attributes

    as_graphs: bool (default: False)
        whether to return the dataset as a list of networkx graphs

    Attributes
    ----------
    metadata: dict
        the metadata of the dataset

    readme: str
        the readme file of the dataset

    data: list
        the data of the dataset, same as the data from of fetch_dataset()

    graphs: list
        the data in the form of networkx.Graph of the dataset, used for SP and GL kernels

    y: list
        the target values of the dataset

    G: networkx.Graph
        the data in the form of networkx of the dataset, used for WL kernel

    node_labels: list
        the node labels of the dataset

    node_color_map: dict
        the color map of the node labels

    edge_labels: list
        the edge labels of the dataset    
    """
    def __init__(self, name, cmap="coolwarm", from_TUDataset=True,
                 verbose=False, data_home=None, download_if_missing=True,
                 with_classes=True, produce_labels_nodes=False,
                 prefer_attr_nodes=False, prefer_attr_edges=False,
                 as_graphs=False):
        
        self.name = name

        if from_TUDataset:
            dataset = fetch_dataset(self.name, verbose=verbose, data_home=data_home,
                                    download_if_missing=download_if_missing,
                                    with_classes=with_classes,
                                    produce_labels_nodes=produce_labels_nodes,
                                    prefer_attr_nodes=prefer_attr_nodes,
                                    prefer_attr_edges=prefer_attr_edges,
                                    as_graphs=as_graphs)
        else:
            dataset = read_data(self.name, with_classes=with_classes,
                                prefer_attr_nodes=prefer_attr_nodes,
                                prefer_attr_edges=prefer_attr_edges,
                                produce_labels_nodes=produce_labels_nodes,
                                as_graphs=as_graphs,
                                is_symmetric=False)
            
        self.metadata = dataset.metadata
        self.readme = dataset.readme
        self.data = dataset.data
    
        try:
            self.y = dataset.target

            # for MUTAG dataset
            if -1 in set(self.y) or 0 in set(self.y):
                self.y = [0 if y == 1 else 1 for y in self.y]
                    
            # for other datasets
            # elif 0 not in set(self.dataset.y):
            #     self.dataset.y = [y-1 for y in self.dataset.y]
        except:
            print("The dataset does not have target values.")

        self.graphs = []
        self.G = nx.Graph()
        
        # set node color map
        if self.metadata[name]["nl"] == True:
            self.node_labels = list(set(label for g in self.data for label in g[1].values()))
            if self.name == "AIDS":
                # AIDS has too many nodes, so it uses two color maps
                node_cmap_1 = plt.get_cmap("tab20", 20)
                node_cmap_2 = plt.get_cmap("tab20b", 18)
                self.node_color_map_1 = {self.node_labels[index]: node_cmap_1(index) for index in range(len(self.node_labels[:20]))}
                self.node_color_map_2 = {self.node_labels[index+20]: node_cmap_2(index) for index in range(len(self.node_labels[20:]))}
                self.node_color_map = {**self.node_color_map_1, **self.node_color_map_2}
            else:
                node_cmap = plt.get_cmap(cmap, len(self.node_labels))
                self.node_color_map = {self.node_labels[index]: node_cmap(index) for index in range(len(self.node_labels))}

            for g in self.data:
                nx_G = nx.Graph()
                for node in g[1].items():
                    self.G.add_node(node[0], label=node[1])
                    nx_G.add_node(node[0], label=node[1])
                self.graphs.append(nx_G)

            if self.metadata[name]["el"] == True:
                self.edge_labels = list(set(label for g in self.data for label in g[2].values()))
                for g, graph in zip(self.data, self.graphs):
                    for edge in g[2].items():
                        self.G.add_edge(edge[0][0], edge[0][1], type=edge[1])
                        graph.add_edge(edge[0][0], edge[0][1], type=edge[1])
            
            else:
                for g, graph in zip(self.data, self.graphs):
                    for edge in g[0]:
                        self.G.add_edge(edge[0], edge[1])
                        graph.add_edge(edge[0], edge[1])
                        
        else:
            if self.metadata[name]["el"] == True:
                self.edge_labels = list(set(label for g in self.data for label in g[2].values()))
                for g in self.data:
                    nx_G = nx.Graph()
                    for edge in g[2].items():
                        self.G.add_edge(edge[0][0], edge[0][1], type=edge[1])
                        nx_G.add_edge(edge[0][0], edge[0][1], type=edge[1])
                    self.graphs.append(nx_G)
                        
            else:
                for g in self.data:
                    nx_G = nx.Graph()
                    for edge in g[0]:
                        self.G.add_edge(edge[0], edge[1])
                        nx_G.add_edge(edge[0], edge[1])
                    self.graphs.append(nx_G)

    def set_color_map(self, cmap):
        if self.name == "AIDS":
            print("The color map of ADIS dataset is fixed.")
        else:
            if self.metadata[self.name]["nl"] == True:
                node_cmap = plt.get_cmap(cmap, len(self.node_labels))
                self.node_color_map = {self.node_labels[index]: node_cmap(index) for index in range(len(self.node_labels))}
            else:
                raise ValueError("The dataset does not have node labels.")
    
    def plot_graph(self, index, node_size=80, with_labels=False, node_feature_color = None, edge_color="k", pos = None, figsize=(5, 5)):
        if self.metadata[self.name]["nl"] == True:
            node_color = [self.node_color_map[label[1]] for label in self.graphs[index].nodes(data="label")]
        else:
            node_color = "tab:blue"

        if self.metadata[self.name]["el"] == True:
            # +0.5 for edge width, because edge width 0 is not visible
            edge_width = [type[2]+0.5 for type in self.graphs[index].edges(data="type")]
        else:
            edge_width = 0.5

        plt.figure(figsize=figsize, dpi=200)
        plt.margins(0.0)
        if pos:
            # for plot individual graph with if the kernel is graphlet
            nx.draw(self.graphs[index], pos=pos, node_color=node_color, node_size=node_size, edge_color = edge_color, width=edge_width, with_labels=with_labels)
        else:
            pos = nx.nx_agraph.pygraphviz_layout(self.graphs[index])
            if node_feature_color:
                nx.draw(self.graphs[index], pos=pos, node_color=node_feature_color, node_size=node_size, edge_color = edge_color, width=edge_width, with_labels=with_labels)
            else:
                nx.draw(self.graphs[index], pos=pos, node_color=node_color, node_size=node_size, edge_color = edge_color, width=edge_width, with_labels=with_labels)
        plt.show()

    def plot_G(self, node_size=30, figsize=(50, 50)):
        if self.metadata[self.name]["nl"] == True:
            node_color = [self.node_color_map[label[1]] for label in self.G.nodes(data="label")]
        else:
            node_color = "tab:blue"

        if self.metadata[self.name]["el"] == True:
            # +0.5 for edge width, because edge width 0 is not visible
            edge_width = [type[2]+0.5 for type in self.G.edges(data="type")]
        else:
            edge_width = 0.5

        plt.figure(figsize=figsize, dpi=100)
        plt.margins(-0.03)
        pos = nx.nx_agraph.pygraphviz_layout(self.G)
        nx.draw(self.G, pos=pos, node_color=node_color, width=edge_width, node_size=node_size)
        plt.show()

    def plot_class_distribution(self, with_figures=True):
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            counter = Counter(self.y)
            bars = ax.bar(counter.keys(), counter.values(), width=0.4)
            if with_figures:
                for bars in ax.containers:
                    # make the numbers on bars away from the bars
                    ax.bar_label(bars, padding=3)
            else:
                pass
        except ValueError:
            print("The dataset does not have target values.")

        # give each bar a different color, using colormap
        cmap = plt.get_cmap("coolwarm", len(counter))
        for i, bar in enumerate(bars):
            bar.set_color(cmap(i))

        # make the y-axis larger to show the numbers on bars
        ax.margins(y=0.1)
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of samples")
        ax.set_title(self.name + " dataset")
        plt.show()

    def plot_color_map(self):
        if self.metadata[self.name]["nl"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            for key, color in self.node_color_map.items():
                ax.bar(key, 1, color=color)

            ax.get_yaxis().set_visible(False)

            plt.show()
        else:
            raise ValueError("The dataset does not have node labels.")