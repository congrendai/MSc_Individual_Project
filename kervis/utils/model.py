import shap
import scipy
import numpy as np
import networkx as nx
from sklearn.svm import SVC
from itertools import combinations
from matplotlib import pyplot as plt
from kervis.utils.dataset import Dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from kervis.kernels import VertexHistogram, EdgeHistogram, ShortestPath, Graphlet, WeisfeilerLehman

class Model:
    def __init__(self, kernel, dataset, model, test_size=0.2, shuffle=False, seed=None, camp = "coolwarm"):
        self.kernel = kernel
        self.seed = seed
        self.dataset = dataset
        if type(self.kernel) == type(VertexHistogram()) or type(self.kernel) == type(EdgeHistogram()):
            self.kernel.fit_transform(self.dataset.data) 
        else:
            self.kernel.fit_transform(self.dataset.graphs)    
        
        self.features = self.kernel.X
        
        if type(self.features) == scipy.sparse.csr.csr_matrix:
            self.features= self.features.toarray()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.dataset.y, test_size=test_size, shuffle=shuffle)

        if model == 'SVM':
            self.clf = SVC(kernel='rbf', gamma='auto')
            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)

        elif model == 'AdaBoost':
            self.clf = AdaBoostClassifier(n_estimators=100, random_state=0)
            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)

    def explain(self, algorithm="auto"):
        if algorithm == "permutation":
            # Use SHAP to explain the model's predictions
            self.explainer = shap.Explainer(self.clf.predict, self.X_train, algorithm="permutation", seed=self.seed, max_evals=2*self.X_train.shape[1]+1)
        else:
            self.explainer = shap.Explainer(self.clf.predict, self.X_train, algorithm=algorithm, seed=self.seed)
        
        self.shap_values = self.explainer(self.X_test)

    def find_features(self, graph_index, shap_feature_index):
        if type(self.kernel) == type(VertexHistogram()):
            return [key for key, value in self.dataset.graphs[graph_index].nodes(data="label") if value == self.kernel.attributes[shap_feature_index]]

        elif type(self.kernel) == type(EdgeHistogram()):
            return [(u, v) for u, v, t in self.dataset.graphs[graph_index].edges(data="type") if t == self.kernel.attributes[shap_feature_index]]
        
        elif type(self.kernel) == type(Graphlet()):
            graphlets_in_graph = []
            graph = self.dataset.graphs[graph_index]
            C = combinations(list(graph), self.kernel.k)
            for c in C:
                if nx.is_isomorphic(graph.subgraph(c), self.kernel.graphlets[self.kernel.attributes[shap_feature_index]]):
                    graphlets_in_graph.append(c)

            return graphlets_in_graph

        elif type(self.kernel) == type(ShortestPath()):
            paths = []
            for path in nx.all_pairs_shortest_path_length(self.dataset.graphs[graph_index]):
                for key, value in path[1].items():
                    if value == self.kernel.attributes[shap_feature_index][2]:
                        paths.append((*sorted((path[0], key)), value))

            paths = list(set(paths))
            paths_in_graph = []
            for path in paths:
                if self.dataset.graphs[graph_index].nodes(data="label")[path[0]] == self.kernel.attributes[shap_feature_index][0] \
                    and self.dataset.graphs[graph_index].nodes(data="label")[path[1]] == self.kernel.attributes[shap_feature_index][1]:
                    paths_in_graph.append(path)

            return paths_in_graph

        

        elif type(self.kernel) == type(WeisfeilerLehman()):
            pass

    def highlight_features(self, graph_index, shap_feature_index, node_size = 80, figsize=10, with_labels=False, all=True):
        features = self.find_features(graph_index, shap_feature_index)
        
        if features:
            if all:
                graph = self.dataset.graphs[graph_index]
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
                    for key, value in self.dataset.graphs[graph_index].nodes(data="label"):
                        if key in features:
                            node_color.append((1,0,0,1))
                        else:
                            node_color.append(self.dataset.node_color_map[value])
                    self.dataset.plot_graph(graph_index, node_feature_color=node_color, with_labels=with_labels, node_size=node_size)

                elif type(self.kernel) == type(EdgeHistogram()):
                    edge_color = ['r' if edge in features else 'k' for edge in self.dataset.graphs[graph_index].edges()]
                    self.dataset.plot_graph(graph_index, edge_color=edge_color, with_labels=with_labels, node_size=node_size)

                elif type(self.kernel) == type(Graphlet()):
                    features = [node+i*len(nodes) for i, feature in enumerate(features) for node in feature]
                    plt.figure(figsize=(figsize, figsize), dpi=100)
                    plt.margins(0.0)
                    ax = nx.draw(G, pos=pos, node_color=node_color, width=edge_width, node_size=node_size)
                    nx.draw(G.subgraph(features), pos=pos, node_color="r", node_size=node_size, edge_color="r", width=edge_width, with_labels=with_labels, ax=ax)

                elif type(self.kernel) == type(ShortestPath()):
                    features = [np.array(feature)+i*len(nodes) for i, feature in enumerate(features)]
                    edge_color_index = []
                    for feature in features:
                        path = nx.shortest_path(G, source=feature[0], target=feature[1])
                        for index, edge in enumerate(G.edges()):
                            for i in range(len(path)-1):
                                if (path[i], path[i+1]) == edge or (path[i+1], path[i]) == edge:
                                    edge_color_index.append(index)

                    edge_color = ['r' if index in edge_color_index else 'k' for index in range(len(G.edges()))]
                    plt.figure(figsize=(figsize, figsize), dpi=100)
                    plt.margins(0.0)
                    nx.draw(G, pos=pos, node_color=node_color, edge_color=edge_color, width=edge_width, node_size=node_size, with_labels=with_labels)

                elif type(self.kernel) == type(WeisfeilerLehman()):
                    pass
            else:
                pos = nx.nx_agraph.pygraphviz_layout(self.dataset.graphs[graph_index])
                if type(self.kernel) == type(ShortestPath()):
                    for feature in features:
                        path = nx.shortest_path(self.dataset.graphs[graph_index], source=feature[0], target=feature[1])
                        edge_color_index = []
                        for index, edge in enumerate(self.dataset.graphs[graph_index].edges()):
                            for i in range(len(path)-1):
                                if (path[i], path[i+1]) == edge or (path[i+1], path[i]) == edge:
                                    edge_color_index.append(index)

                        edge_color = ['r' if index in edge_color_index else 'k' for index in range(len(self.dataset.graphs[graph_index].edges()))]
                        self.dataset.plot_graph(graph_index, edge_color=edge_color, with_labels=with_labels, node_size=node_size)

                elif type(self.kernel) == type(Graphlet()):
                    graph = self.dataset.graphs[graph_index]
                    edge_width = [type[2]+2 for type in graph.edges(data="type")]
                    pos = nx.nx_agraph.pygraphviz_layout(graph)
                    for feature in features:
                        ax = self.dataset.plot_graph(graph_index, graphlet_pos=pos, node_size=node_size, with_labels=with_labels)
                        nx.draw(graph.subgraph(feature), pos=pos, node_color="r", node_size=node_size, edge_color="r", width=edge_width, with_labels=with_labels, ax=ax)
                    
        else:
            print("No feature found in graph {}".format(graph_index))

    # SHAP plots
    def summary_plot(self, max_display=20):
        shap.plots.beeswarm(self.shap_values, max_display=max_display)

    def force_plot(self, graph_index):
        shap.plots.force(self.shap_values[graph_index], matplotlib=True)

    def bar_plot(self, graph_index, max_display=None):
        shap.bar_plot(self.shap_values.values[graph_index], max_display=max_display)

    def waterfall_plot(self, graph_index, max_display=10):
        shap.plots.waterfall(self.shap_values[graph_index], max_display=max_display)

    def heatmap_plot(self, max_display=10):
        shap.plots.heatmap(self.shap_values, max_display=max_display)