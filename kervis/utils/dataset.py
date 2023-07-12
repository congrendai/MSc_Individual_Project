import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
from kervis.utils.utils import fetch_dataset

class Dataset():
    def __init__(self, name, cmap):
        self.name = name
        dataset = fetch_dataset(self.name, verbose=False)
        self.metadata = dataset.metadata
        self.readme = dataset.readme
        self.data = dataset.data
    
        try:
            self.y = dataset.target
        except:
            print("The dataset does not have target values.")

        self.graphs = []
        self.G = nx.Graph()
        
        if self.metadata[name]["nl"] == True:
            self.node_labels = list(set(label for g in self.data for label in g[1].values()))
            if self.name == "AIDS":
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
            
    
    def plot_graph(self, index, node_size=80, with_labels=False, node_feature_color = None, edge_color="k", graphlet_pos = None):
        if self.metadata[self.name]["nl"] == True:
            node_color = [self.node_color_map[label[1]] for label in self.graphs[index].nodes(data="label")]
        else:
            node_color = "tab:blue"

        if self.metadata[self.name]["el"] == True:
            # +0.5 for edge width, because edge width 0 is not visible
            edge_width = [type[2]+0.5 for type in self.graphs[index].edges(data="type")]
        else:
            edge_width = 0.5

        plt.figure(figsize=(5, 5), dpi=100)
        plt.margins(0.0)
        if graphlet_pos:
            nx.draw(self.graphs[index], pos=graphlet_pos, node_color=node_color, node_size=node_size, edge_color = edge_color, width=edge_width, with_labels=with_labels)
        else:
            pos = nx.nx_agraph.pygraphviz_layout(self.graphs[index])
            if node_feature_color:
                nx.draw(self.graphs[index], pos=pos, node_color=node_feature_color, node_size=node_size, edge_color = edge_color, width=edge_width, with_labels=with_labels)
            else:
                nx.draw(self.graphs[index], pos=pos, node_color=node_color, node_size=node_size, edge_color = edge_color, width=edge_width, with_labels=with_labels)

    def plot_G(self, node_size=30):
        if self.metadata[self.name]["nl"] == True:
            node_color = [self.node_color_map[label[1]] for label in self.G.nodes(data="label")]
        else:
            node_color = "tab:blue"

        if self.metadata[self.name]["el"] == True:
            # +0.5 for edge width, because edge width 0 is not visible
            edge_width = [type[2]+0.5 for type in self.G.edges(data="type")]
        else:
            edge_width = 0.5

        plt.figure(figsize=(50, 50), dpi=300)
        plt.margins(-0.03)
        pos = nx.nx_agraph.pygraphviz_layout(self.G)
        nx.draw(self.G, pos=pos, node_color=node_color, width=edge_width, node_size=node_size)

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