import shap
import scipy
import networkx as nx
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from kervis.utils.dataset import Dataset
from sklearn.model_selection import train_test_split
from kervis.kernels import VertexHistogram, EdgeHistogram, ShortestPath, Graphlet, WeisfeilerLehman

class Model:
    def __init__(self, dataset_name, kernel , model, test_size=0.2, shuffle=False):
        self.kernel = kernel()
        self.dataset = Dataset(dataset_name)
        if type(self.kernel) == type(VertexHistogram()) or type(self.kernel) == type(EdgeHistogram()):
            self.kernel.fit_transform(self.dataset.data) 
        else:
            self.kernel.fit_transform(self.dataset.graphs)    
        
        self.features = self.kernel.X
        
        if type(self.features) == scipy.sparse.csr.csr_matrix:
            self.features= self.features.toarray()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.dataset.y, test_size=test_size, shuffle=shuffle)

        if model == 'SVM':
            self.clf = SVC(kernel='linear')
            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)

        # Use SHAP to explain the model's predictions
        self.explainer = shap.Explainer(self.clf.predict, self.X_train)
        self.shap_values = self.explainer(self.X_test)

    def find_features(self, graph_index, shap_feature_index):
        if type(self.kernel) == type(VertexHistogram()):
            return [key for key, value in self.dataset.graphs[graph_index].nodes(data="label") if value == self.kernel.attributes[shap_feature_index]]

        elif type(self.kernel) == type(EdgeHistogram()):
            return [(u, v) for u, v, t in self.dataset.graphs[graph_index].edges(data="type") if t == self.kernel.attributes[shap_feature_index]]

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

        elif type(self.kernel) == type(Graphlet()):
            return self.kernel.attributes[shap_feature_index]

        elif type(self.kernel) == type(WeisfeilerLehman()):
            pass

    def highlight_features(self, graph_index, shap_feature_index):
        features = self.find_features(graph_index, shap_feature_index)
        if features:
            pos = nx.nx_agraph.pygraphviz_layout(self.dataset.graphs[graph_index])

            if type(self.kernel) == type(VertexHistogram()):
                node_color = []
                for key, value in self.dataset.graphs[graph_index].nodes(data="label"):
                    if key in features:
                        node_color.append((1,0,0,1))
                    else:
                        node_color.append(self.dataset.node_color_map[value])
                self.dataset.plot_graph(graph_index, node_feature_color=node_color)

            elif type(self.kernel) == type(EdgeHistogram()):
                edge_color = ['red' if edge in features else 'black' for edge in self.dataset.graphs[graph_index].edges()]
                self.dataset.plot_graph(graph_index, edge_color=edge_color)

            elif type(self.kernel) == type(ShortestPath()):
                pass

            elif type(self.kernel) == type(Graphlet()):
                pass

            elif type(self.kernel) == type(WeisfeilerLehman()):
                pass
        else:
            print("No feature found in graph {}".format(graph_index))

    # SHAP plots
    def summary_plot(self):
        shap.summary_plot(self.shap_values)

    def force_plot(self, sample_index):
        shap.force_plot(self.shap_values[sample_index], matplotlib=True)

    def bar_plot(self, sample_index):
        shap.bar_plot(self.shap_values.values[sample_index])

    def waterfall_plot(self, sample_index):
        shap.plots.waterfall(self.shap_values[sample_index])

    def heatmap_plot(self):
        shap.plots.heatmap(self.shap_values)