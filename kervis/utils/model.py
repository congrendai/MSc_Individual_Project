import shap
import scipy
import numpy as np
import pandas as pd
import xgboost as xgb
import networkx as nx
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from itertools import combinations
from matplotlib import pyplot as plt
from kervis.utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from kervis.kernels import VertexHistogram, EdgeHistogram, ShortestPath, Graphlet, WeisfeilerLehman

class Model:
    def __init__(self, kernel, dataset, model, test_size=0.2, shuffle=False, seed=None, camp = "coolwarm"):
        self.kernel = kernel
        self.seed = seed
        self.dataset = dataset

        if type(self.kernel) == type(VertexHistogram()) or type(self.kernel) == type(EdgeHistogram()) or type(self.kernel) == type(WeisfeilerLehman()):
            self.kernel.fit_transform(self.dataset.data) 
        
        else:
            self.kernel.fit_transform(self.dataset.graphs)    
        
        self.features = self.kernel.X
            
        if type(self.features) == scipy.sparse.csr_matrix:
            self.features= self.features.toarray()

        if -1 in set(self.dataset.y):
                self.dataset.y = [y if y == 1 else 0 for y in self.dataset.y]
                
        elif 0 not in set(self.dataset.y):
            self.dataset.y = [y-1 for y in self.dataset.y]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.dataset.y, test_size=test_size, shuffle=shuffle)

        if model == 'kmeans':
            self.clf = KMeans(n_init="auto", n_clusters=len(set(self.dataset.y)))
            self.clf.fit(self.X_train)
            self.y_pred = self.clf.predict(self.X_test)
        
        elif model == 'svm':
            self.clf = SVC(kernel='rbf', gamma='auto')
            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)

        elif model == 'xgboost':
            self.clf = xgb.XGBClassifier()
            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)
        
        else:
            raise ValueError('Model must be "svm", "kmeans", or "xgboost".')

    def evaluate(self):
        self.evaluator = Evaluator(self)
        self.evaluator.classification_report()

    def explain(self):
        # Use SHAP to explain the model's predictions
        self.explainer = shap.Explainer(self.clf.predict, self.X_train, algorithm="permutation", seed=self.seed, max_evals=2*self.X_train.shape[1]+1)
        self.shap_values = self.explainer(self.X_test)


    def find_features(self, graph_index, shap_feature_index):
        index = len(self.X_train) + graph_index
        if type(self.kernel) == type(VertexHistogram()):
            return [key for key, value in self.dataset.graphs[index].nodes(data="label") if value == self.kernel.attributes[shap_feature_index]]

        elif type(self.kernel) == type(EdgeHistogram()):
            return [(u, v) for u, v, t in self.dataset.graphs[index].edges(data="type") if t == self.kernel.attributes[shap_feature_index]]
        
        elif type(self.kernel) == type(Graphlet()):
            graphlets_in_graph = []
            graph = self.dataset.graphs[index]
            C = combinations(list(graph), self.kernel.k)
            for c in C:
                if nx.is_isomorphic(graph.subgraph(c), self.kernel.graphlets[self.kernel.attributes[shap_feature_index]]):
                    graphlets_in_graph.append(c)

            return graphlets_in_graph

        elif type(self.kernel) == type(ShortestPath()):
            graph = self.dataset.graphs[index]
            nodes = list(graph.nodes)
            shortest_paths = []
            for i in range(len(nodes)-1):
                for j in range(i+1, len(nodes)):
                    for path in nx.all_shortest_paths(graph, nodes[i], nodes[j]):
                        shortest_paths.append(path)

            shortest_paths = [path for path in shortest_paths if len(path)-1 == self.kernel.attributes[shap_feature_index][2]]
            paths_in_graph = []
            for shortest_path in shortest_paths:
                if graph.nodes(data="label")[shortest_path[0]] == self.kernel.attributes[shap_feature_index][0] \
                    and graph.nodes(data="label")[shortest_path[-1]] == self.kernel.attributes[shap_feature_index][1] or \
                        graph.nodes(data="label")[shortest_path[0]] == self.kernel.attributes[shap_feature_index][1] \
                            and graph.nodes(data="label")[shortest_path[-1]] == self.kernel.attributes[shap_feature_index][0]:
                    paths_in_graph.append(shortest_path)

            return paths_in_graph
        
        elif type(self.kernel) == type(WeisfeilerLehman()):
            print("No need to find the feature vectors of the Weisfeiler-Lehman kernel")

    def highlight_features(self, graph_index, shap_feature_index, node_size = 80, sub_edge_width = 3, figsize=10, with_labels=False, all=True):
        features = self.find_features(graph_index, shap_feature_index)
        if features:
            index = len(self.X_train) + graph_index
            graph = self.dataset.graphs[index]
            if all:
                nodes = np.array(graph.nodes())
                edges = np.array(graph.edges())
                graph_color_map = [self.dataset.node_color_map[label[1]] for label in graph.nodes(data="label")]
                node_color = [color for _ in range(len(features)) for color in graph_color_map]
                edge_width = [type[2]+2 for _ in range(len(features)) for type in graph.edges(data="type")]
                G = nx.Graph()
                for i in range(len(features)):  
                        G.add_nodes_from(nodes+i*len(nodes))
                        G.add_edges_from(edges+i*len(nodes))
                pos = nx.nx_agraph.pygraphviz_layout(G)

                if type(self.kernel) == type(VertexHistogram()):
                    node_color = []
                    for key, value in self.dataset.graphs[index].nodes(data="label"):
                        if key in features:
                            node_color.append(self.dataset.node_color_map[value])
                        else:
                            node_color.append((0,0,0,0))

                    self.dataset.plot_graph(index, node_feature_color=node_color, with_labels=with_labels, node_size=node_size)

                elif type(self.kernel) == type(EdgeHistogram()):
                    edge_color = ['r' if edge in features else 'k' for edge in self.dataset.graphs[index].edges()]
                    self.dataset.plot_graph(index, edge_color=edge_color, with_labels=with_labels, node_size=node_size)

                elif type(self.kernel) == type(Graphlet()):
                    features = [node+i*len(nodes) for i, feature in enumerate(features) for node in feature]
                    plt.figure(figsize=(figsize, figsize), dpi=100)
                    plt.margins(0.0)
                    ax = nx.draw(G, pos=pos, node_color=node_color, width=edge_width, node_size=node_size)
                    nx.draw(G.subgraph(features), pos=pos, node_color="r", node_size=node_size, width=sub_edge_width, edge_color="r", with_labels=with_labels, ax=ax)

                elif type(self.kernel) == type(ShortestPath()):
                    features = [np.array(feature)+i*len(nodes) for i, feature in enumerate(features)]
                    features = [node for feature in features for node in feature]
                    plt.figure(figsize=(figsize, figsize), dpi=100)
                    plt.margins(0.0)
                    nx.draw(G, pos=pos, node_color=node_color, width=edge_width, node_size=node_size, with_labels=with_labels)
                    nx.draw_networkx_edges(G.subgraph(features), pos=pos, edge_color="r", width=sub_edge_width)

                elif type(self.kernel) == type(WeisfeilerLehman()):
                    print("Cannot highlight the Weisfeiler-Lehman kernel")
            else:
                pos = nx.nx_agraph.pygraphviz_layout(graph)
                if type(self.kernel) == type(ShortestPath()):
                    for feature in features:
                        ax = self.dataset.plot_graph(index, node_size=node_size, with_labels=with_labels)
                        nx.draw_networkx_edges(graph.subgraph(feature), pos=pos, edge_color="r", width=sub_edge_width, ax=ax)

                elif type(self.kernel) == type(Graphlet()):
                    edge_width = [type[2]+2 for type in graph.edges(data="type")]
                    for feature in features:
                        ax = self.dataset.plot_graph(index, pos=pos, node_size=node_size, with_labels=with_labels)
                        nx.draw(graph.subgraph(feature), pos=pos, node_color="r", node_size=node_size, edge_color="r", width=edge_width, with_labels=with_labels, ax=ax)
                    
        else:
            print("No feature found in graph {}".format(graph_index))

    def highlight_all(self, feature_index, node_size = 80, figsize=10, with_labels=False):
        features = [self.find_features(graph_index, feature_index) for graph_index in range(len(self.X_test))]
        pass


    # SHAP plots
    def summary_plot(self, max_display=20):
        shap.plots.beeswarm(self.shap_values, max_display=max_display)

    def force_plot(self, graph_index):
        shap.plots.force(self.shap_values[graph_index], matplotlib=True)

    def bar_plot(self, graph_index=None, max_display=None):
        if graph_index == None:
            shap.plots.bar(self.shap_values, max_display=max_display)
        else:
            shap.plots.bar(self.shap_values[graph_index], max_display=max_display)

    def waterfall_plot(self, graph_index, max_display=10):
        shap.plots.waterfall(self.shap_values[graph_index], max_display=max_display)

    def heatmap_plot(self, max_display=10):
        shap.plots.heatmap(self.shap_values, max_display=max_display)