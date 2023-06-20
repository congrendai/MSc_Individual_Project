import networkx as nx
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
from utils import fetch_dataset
import streamlit.components.v1 as components
from pyvis.network import Network

class Graph():
    def __init__(self, g, node_color_map, with_node_labels, with_egde_labels):
        self.graph = nx.Graph()
        self.node_color_map = node_color_map
        self.with_node_labels = with_node_labels
        self.with_egde_labels = with_egde_labels
        # add nodes and labels, if labels are available
        if self.with_node_labels:
            for i in g[1].items():
                self.graph.add_node(i[0], label=i[1])

        if self.with_egde_labels:
        # add edges and labels, if labels are available
            for i in g[2].items():
                self.graph.add_edge(i[0][0], i[0][1], type=i[1])
        else:
            for i in g[0]:
                self.graph.add_edge(i[0], i[1])

    def plot(self, with_labels=False):
        if len(self.node_color_map) == 0:
            node_color = "gray"
            
        else:
            node_color = [self.node_color_map[label[1]] for label in self.graph.nodes(data="label")]

        if self.with_egde_labels:
            # +1 for edge width, because edge width 0 is not visible
            edge_width = [self.graph[u][v]["type"] + 1 for u, v in self.graph.edges()]
            
        else:
            edge_width = 1

        plot_graph(self.graph)

        # fig, ax = plt.subplots(figsize=(5, 5))
        # pos = nx.spring_layout(self.graph)
        # nx.draw(self.graph, pos=pos, node_color=node_color, with_labels=with_labels, node_size=80, width=edge_width, edge_color="gray")
        # st.pyplot(fig)

class Dataset():
    def __init__(self, name):
        self.name = name
        self.with_node_labels = True
        self.with_edge_labels = True
        self.dataset = fetch_dataset(self.name, verbose=False)
        self.readme = self.dataset.readme
        self.G, self.y = self.dataset.data, self.dataset.target

        # Some datasets have no node labels, e.g. FRANKENSTEIN
        try:
            assert len(self.G[0][1]) != 0
            self.nodes = [node for g in self.G for node in g[1].items()]
            self.node_labels = list(set(label for g in self.G for label in g[1].values()))
            node_cmap = plt.get_cmap("tab20")
            self.node_color_map = {self.node_labels[index]: node_cmap.colors[index] for index in range(len(self.node_labels))}

        except AssertionError:
            self.nodes = list(set([node for g in self.G for edge in g[0] for node in edge]))
            self.with_node_labels = False
            self.node_color_map = {}
            print("{} nodes have no labels.".format(name))
        

        # Some datasets have no edge labels, e.g. BZR
        try:
            assert len(self.G[0][2]) != 0
            self.edges = [edge for g in self.G for edge in g[2].items()]
            self.edge_labels = list(set(label for g in self.G for label in g[2].values()))

        except AssertionError:
            self.edges = [edge for g in self.G for edge in g[0]]
            self.with_edge_labels = False
            print("{} edges have no labels.".format(name))

        self.graphs = [Graph(g, self.node_color_map, self.with_node_labels, self.with_edge_labels) for g in self.G]

    def get_readme(self):
        return self.readme

    def plot_dataset(self):
        nx_G = nx.Graph()

        if self.with_node_labels:
            for node in self.nodes:
                nx_G.add_node(node[0], label=node[1])
                self_node_color = [self.node_color_map[label[1]] for label in nx_G.nodes(data="label")]

        else:
            for node in self.nodes:
                nx_G.add_node(node)
                self_node_color = "gray"

        if self.with_edge_labels:
            for edge in self.edges:
                nx_G.add_edge(edge[0][0], edge[0][1], type=edge[1])
                self.edge_width = [nx_G[u][v]["type"] + 1 for u, v in nx_G.edges()]

        else:
            for edge in self.edges:
                nx_G.add_edge(edge[0], edge[1])
                self.edge_width = 1

        fig, ax = plt.subplots(figsize=(5, 5))
        pos = nx.spring_layout(nx_G)
        nx.draw(nx_G, pos=pos, node_color=self_node_color, with_labels=False, node_size=100, width=self.edge_width, edge_color="gray")
        st.pyplot(fig)

        return nx_G

    def plot_class_distribution(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        counter = Counter(self.y)
        bars = ax.bar(counter.keys(), counter.values(), width=0.4)
        for bars in ax.containers:
            # make the numbers on bars away from the bars
            ax.bar_label(bars, padding=3)

        # give each bar a different color, using colormap
        cmap = plt.get_cmap("tab20")
        for i, bar in enumerate(bars):
            bar.set_color(cmap(i))

        # make the y-axis larger to show the numbers on bars
        ax.margins(y=0.1)
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of samples")
        ax.set_title(self.name + " dataset")
        st.pyplot(fig)

def plot_graph(graph):
    graph_net = Network()

    graph_net.from_nx(graph)

    graph_net.repulsion(node_distance=420, central_gravity=0.33,
                        spring_length=110, spring_strength=0.10,
                        damping=0.95)

    try:
        path = '/tmp'
        graph_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        path = '/html_files'
        graph_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=435)