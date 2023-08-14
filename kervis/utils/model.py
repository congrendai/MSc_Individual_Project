import shap
import scipy
import numpy as np
import xgboost as xgb
import networkx as nx
from sklearn.svm import SVC
from collections import Counter
from itertools import combinations
from matplotlib import pyplot as plt
from kervis.utils.evaluator import Evaluator
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from kervis.kernels import VertexHistogram, EdgeHistogram, ShortestPath, Graphlet, GraphletSampling, WeisfeilerLehman

class Model:
    """
    This class is combined with the Kernel class and Dataset class to build a model

    Parameters
    ----------
    kernel: kervis.kernels.Kernel
        the kernel to be used

    dataset: kervis.datasets.Dataset
        the dataset to be used

    model: str
        the model to be used

    test_size: float (default: 0.2)
        the size of the test set

    shuffle: bool (default: False)
        whether to shuffle the dataset before splitting

    seed: int (default: None)
        the seed for SHAP estimates Shapley values

    camp: str (default: "coolwarm")
        the color map for node colors

    Attributes
    ----------
    kernel: kervis.kernels.Kernel
        the kernel to be used

    dataset: kervis.datasets.Dataset
        the dataset to be used

    features: numpy.ndarray
        the feature matrix

    name: str
        the name of the model, which is the combination of the kernel name, dataset name, and method name

    X_train: numpy.ndarray
        the feature matrix of the training set

    X_test: numpy.ndarray
        the feature matrix of the test set

    y_train: numpy.ndarray
        the labels of the training set
    
    y_test: numpy.ndarray
        the labels of the test set

    clf: sklearn.svm.SVC or sklearn.linear_model.LogisticRegression or xgboost.XGBClassifier
        the classifier to be used

    y_pred: numpy.ndarray
        the predicted labels of the test set

    cv_scores: numpy.ndarray
        the cross validation scores

    evaluator: kervis.utils.evaluator.Evaluator
        contains the evaluation results

    explainer: shap.Explainer
        the "permutation" explainer from SHAP

    shap_values: numpy.ndarray
        the Shapley values of the test set
    """

    def __init__(self, kernel, dataset, model, test_size=0.2, shuffle=False, seed=None, camp = "coolwarm"):
        self.kernel = kernel
        self.seed = seed
        self.dataset = dataset
        self.name = "{} {}".format(self.kernel.name, self.dataset.name)

        if type(self.kernel) == type(ShortestPath()) or type(self.kernel) == type(Graphlet()):
            self.kernel.fit_transform(self.dataset.graphs) 
        
        else:
            self.kernel.fit_transform(self.dataset.data)    
        
        if type(self.kernel) == type(GraphletSampling()):
            self.features = self.kernel._phi_X
        else:
            self.features = self.kernel.X
            
        if type(self.features) == scipy.sparse.csr_matrix:
            self.features = self.features.toarray()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.dataset.y, test_size=test_size, shuffle=shuffle)

        if model == 'logistic':
            self.clf = LogisticRegression(max_iter=200000)
        
        elif model == 'svm':
            self.clf = SVC(kernel='rbf', gamma='auto')

        elif model == 'xgboost':
            self.clf = xgb.XGBClassifier()
        
        else:
            raise ValueError('Model must be "svm", "logistic", or "xgboost".')

        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)

    def cross_validate(self, cv=10):
        # for calculate cross-validatopm scores
        self.cv_scores = cross_val_score(self.clf, self.X_train, self.y_train, cv=cv)
    
    def sparsity(self):
        # Calculate the sparsity of the feature matrix
        return 1 - np.count_nonzero(self.X_train) / self.X_train.size

    def evaluate(self):
        # Evaluate the model
        self.evaluator = Evaluator(self)

    def explain(self):
        # Use SHAP to explain the model's predictions
        self.explainer = shap.Explainer(self.clf.predict, self.X_train, algorithm="permutation", seed=self.seed, max_evals=2*self.X_train.shape[1]+1)
        self.shap_values = self.explainer(self.X_test)

    def find_features(self, graph_index, shap_feature_index):
        index = graph_index
        if type(self.kernel) == type(VertexHistogram()):
            return [key for key, value in self.dataset.graphs[index].nodes(data="label") if value == self.kernel.attributes[shap_feature_index]]

        elif type(self.kernel) == type(EdgeHistogram()):
            return [(u, v) for u, v, t in self.dataset.graphs[index].edges(data="type") if t == self.kernel.attributes[shap_feature_index]]
        
        elif type(self.kernel) == type(GraphletSampling()):
            graphlets_in_graph = []
            graph = self.dataset.graphs[index]
            C = combinations(list(graph), self.kernel.k)
            for c in C:
                if nx.is_isomorphic(graph.subgraph(c), self.kernel._networkx_graph_bins[self.kernel.attributes[shap_feature_index]]):
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
            print(self.kernel.WL_labels[shap_feature_index])

    def highlight_features(self, graph_index, shap_feature_index, node_size = 80, sub_edge_width = 3, figsize=(10,10), with_labels=False, path="."):
        features = self.find_features(graph_index, shap_feature_index)
        if features:
            index = graph_index
            graph = self.dataset.graphs[index]

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
                for key, value in graph.nodes(data="label"):
                    if key in features:
                        node_color.append(self.dataset.node_color_map[value])
                    else:
                        node_color.append((0,0,0,0))

                self.dataset.plot_graph(index, node_feature_color=node_color, with_labels=with_labels, node_size=node_size, figsize=figsize)
                plt.savefig("{}/{}_{}_{}.png".format(path, self.kernel.name, index, shap_feature_index))
                plt.show()

            elif type(self.kernel) == type(EdgeHistogram()):
                edge_color = ['r' if edge in features else 'k' for edge in self.dataset.graphs[index].edges()]
                self.dataset.plot_graph(index, edge_color=edge_color, with_labels=with_labels, node_size=node_size, figsize=figsize)
                plt.savefig("{}/{}_{}_{}.png".format(path, self.kernel.name, index, shap_feature_index))
                plt.show()

            elif type(self.kernel) == type(GraphletSampling()):
                features = [node+i*len(nodes) for i, feature in enumerate(features) for node in feature]
                plt.figure(figsize=figsize, dpi=300)
                plt.margins(0.0)
                ax = nx.draw(G, pos=pos, node_color=node_color, width=edge_width, node_size=node_size)
                nx.draw(G.subgraph(features), pos=pos, node_color="r", node_size=node_size, width=sub_edge_width, edge_color="r", with_labels=with_labels, ax=ax)
                plt.savefig("{}/{}_{}_{}.png".format(path, self.kernel.name, index, shap_feature_index))
                plt.show()

            elif type(self.kernel) == type(ShortestPath()):
                features = [np.array(feature)+i*len(nodes) for i, feature in enumerate(features)]
                features = [node for feature in features for node in feature]
                plt.figure(figsize=figsize, dpi=300)
                plt.margins(0.0)
                nx.draw(G, pos=pos, node_color=node_color, width=edge_width, node_size=node_size, with_labels=with_labels)
                nx.draw_networkx_edges(G.subgraph(features), pos=pos, edge_color="r", width=sub_edge_width)
                plt.savefig("{}/{}_{}_{}.png".format(path, self.kernel.name, index, shap_feature_index))
                plt.show()

            elif type(self.kernel) == type(WeisfeilerLehman()):
                print("Cannot highlight features of the Weisfeiler-Lehman kernel")
                    
        else:
            print("No feature found in graph {}".format(graph_index))

    def display_T(self, y="P", node_size = 50, figsize=(10,10), with_labels=False):
        base_index = len(self.X_train)

        if y == "p":
            indices = [index for index, y in enumerate(self.y_test) if y == 0]
        elif y == "n":
            indices = [index for index, y in enumerate(self.y_test) if y == 1]
        

        print("The number of {} graphs: {}".format(y, len(indices)))
        print("Graph indices: {}".format(np.array(indices)+base_index))
        graphs = [self.dataset.graphs[index+base_index] for index in indices]
        nodes = [node for graph in graphs for node in graph.nodes()]
        subgraph = nx.subgraph(self.dataset.G, nodes)
        pos = nx.nx_agraph.pygraphviz_layout(subgraph)
        node_color = [self.dataset.node_color_map[node[1]] for node in subgraph.nodes(data="label")]
        width = [type[2]+2 for type in subgraph.edges(data="type")]
        plt.figure(figsize=figsize, dpi=100)
        nx.draw(subgraph, pos=pos, node_color=node_color, width=width, node_size=node_size, with_labels=with_labels)
        plt.savefig("./plots/result/visualization/{}_{}.png".format(self.name, y))
        plt.show()



    def highlight_all(self, y="tp", node_size = 50,  figsize=(10,10), feature=None, critical=True, with_labels=False, path="."):
        base_index = len(self.X_train)

        y_test = np.array(self.y_test)
        y_pred = np.array(self.y_pred)

        
        test_TP = np.where((y_test == 0) & (y_pred == 0))[0] + base_index
        graph_TP = [self.dataset.graphs[index] for index in test_TP]

    
        test_TN = np.where((y_test == 1) & (y_pred == 1))[0] + base_index
        graph_TN = [self.dataset.graphs[index] for index in test_TN]

    
        test_FP = np.where((y_test == 1) & (y_pred == 0))[0] + base_index
        graph_FP = [self.dataset.graphs[index] for index in test_FP]

    
        test_FN = np.where((y_test == 0) & (y_pred == 1))[0] + base_index
        graph_FN = [self.dataset.graphs[index] for index in test_FN]

        if y == "tp":
            nodes = [node for graph in graph_TP for node in graph.nodes()]
            test_graphs = test_TP

        elif y == "tn":
            nodes = [node for graph in graph_TN for node in graph.nodes()]
            test_graphs = test_TN

        elif y == "fp":
            nodes = [node for graph in graph_FP for node in graph.nodes()]
            test_graphs = test_FP

        elif y == "fn":
            nodes = [node for graph in graph_FN for node in graph.nodes()]
            test_graphs = test_FN

        subgraph = nx.subgraph(self.dataset.G, nodes)
        pos = nx.nx_agraph.pygraphviz_layout(subgraph)
        node_color = [self.dataset.node_color_map[node[1]] for node in subgraph.nodes(data="label")]
        width = [type[2]+2 for type in subgraph.edges(data="type")]
        plt.figure(figsize=figsize, dpi=100)

        print("Graph indices: {}".format(np.array(test_graphs)))

        
        shap_indices = [np.argmax(abs(self.shap_values.values[i-base_index])) for i in test_graphs]
        
        print(shap_indices)
        print(Counter(shap_indices))

        if type(self.kernel) == type(VertexHistogram()):
            features = []
            for graph, shap_index in zip(test_graphs, shap_indices):
                features += self.find_features(graph, shap_index)
                
            VH_node_color = []
            for key, value in subgraph.nodes(data="label"):
                if key in features:
                    VH_node_color.append(self.dataset.node_color_map[value])
                else:
                    VH_node_color.append((0,0,0,0))

            nx.draw(subgraph, pos=pos, node_color=VH_node_color, width=width, node_size=node_size, with_labels=with_labels)
            plt.savefig("./plots/result/visualization/{}_{}_critical.png".format(self.name, y))
            plt.show()
            

        elif type(self.kernel) == type(EdgeHistogram()):
            features = []
            for graph, shap_index in zip(test_graphs, shap_indices):
                features += self.find_features(graph, shap_index)

            # edge_color = ['r' if edge in features else 'k' for edge in subgraph.edges()]
            edge_color = []
            for edge in subgraph.edges(data="type"):
                if (edge[0], edge[1]) in features:
                    if edge[2] == 0:
                        edge_color.append("r")
                    elif edge[2] == 1:
                        edge_color.append("g")
                    elif edge[2] == 2:
                        edge_color.append("b")
                    elif edge[2] == 3:
                        edge_color.append("y")
                else:
                    edge_color.append("k")

            nx.draw(subgraph, pos=pos, edge_color=edge_color, width=width, node_color='k', node_size=node_size, with_labels=with_labels)
            plt.savefig("./plots/result/visualization/{}_{}_critical.png".format(self.name, y))
            plt.show()
            

        elif type(self.kernel) == type(GraphletSampling()):
            features = []
            for graph, shap_index in zip(test_graphs, shap_indices):
                if critical:
                    feature = self.find_features(graph, shap_index)
                    # if there are multiple graphlets
                    # only the first one will be highlighted
                    if feature:
                        for node in feature[0]:
                            features.append(node)

                    

                else:
                    self.highlight_features(graph, shap_index, node_size=node_size, figsize=figsize, with_labels=with_labels, path=path)

            nx.draw(subgraph, pos=pos, node_color=node_color, width=width, node_size=node_size, with_labels=with_labels)
            nx.draw(subgraph.subgraph(features), pos=pos, node_color="r", node_size=node_size, edge_color="r", width=width, with_labels=with_labels)
            plt.savefig("./plots/result/visualization/{}_{}_critical.png".format(self.name, y))
            plt.show()

        elif type(self.kernel) == type(ShortestPath()):
            features = []
            for graph, shap_index in zip(test_graphs, shap_indices):
                if critical:
                    feature = self.find_features(graph, shap_index)
                    # if there are multiple shortest paths with the same length
                    # only the first one will be highlighted
                    if feature:
                        for node in feature[0]:
                            features.append(node)
            
                else:
                    self.highlight_features(graph, shap_index, node_size=node_size, figsize=figsize, with_labels=with_labels, path=path)

            nx.draw(subgraph, pos=pos, node_color=node_color, width=width, node_size=node_size, with_labels=with_labels)
            nx.draw_networkx_edges(subgraph.subgraph(features), pos=pos, edge_color="r", width=width)
            plt.savefig("./plots/result/visualization/{}_{}_critical.png".format(self.name, y))
            plt.show()

            

        elif type(self.kernel) == type(WeisfeilerLehman()):
            nx.draw(subgraph, pos=pos, node_color=node_color, width=width, node_size=node_size, with_labels=with_labels)
            plt.show()
            for graph, shap_index in zip(test_graphs, shap_indices):
                self.find_features(graph, shap_index) 

    # SHAP plots
    def summary_plot(self, max_display=20, figsize=None):
        shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
        figure_setting(figsize)

    def force_plot(self, graph_index, figsize=(10,10)):
        shap.plots.force(self.shap_values[graph_index-len(self.X_train)], matplotlib=True, show=False)
        figure_setting(figsize)

    def bar_plot(self, graph_index=None, max_display=None, figsize=None):
        if graph_index == None:
            shap.plots.bar(self.shap_values, max_display=max_display, show=False)
        else:
            shap.plots.bar(self.shap_values[graph_index-len(self.X_train)], max_display=max_display, show=False)
        figure_setting(figsize)


    def waterfall_plot(self, graph_index, max_display=10, figsize=None):
        shap.plots.waterfall(self.shap_values[graph_index-len(self.X_train)], max_display=max_display, show=False)
        figure_setting(figsize)

    def heatmap_plot(self, max_display=10, figsize=None):
        shap.plots.heatmap(self.shap_values, max_display=max_display, show=False)
        figure_setting(figsize)

def figure_setting(figsize):
    if figsize:
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(figsize[0], figsize[1])
    plt.show()