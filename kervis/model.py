import shap
import scipy
from kervis import Dataset
from sklearn.svm import SVC
from kervis.kernels import ShortestPath, Graphlet
from grakel.kernels import VertexHistogram, EdgeHistogram
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, kernel, dataset_name):
        self.kernel = kernel()
        self.dataset = Dataset(dataset_name)
        if self.kernel in [ShortestPath, Graphlet]:
            self.k_dataset = self.kernel.fit_transform(self.dataset.graphs)
        else:
            self.k_dataset = self.kernel.fit_transform(self.dataset.data)    
        
        self.features = self.kernel.X
        
        if type(self.features) == scipy.sparse.csr.csr_matrix:
            self.features= self.features.toarray()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.dataset.target, test_size=0.2, shuffle=False)
        self.clf = SVC(kernel='linear')
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        self.explainer = shap.Explainer(self.clf.predict, self.X_train)
        self.shap_values = self.explainer(self.X_test)
        print("Accuracy for {} is {}".format(dataset_name, accuracy_score(self.y_test, self.y_pred)))

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

    # def summary_plot(self):
    #     st_shap(shap.summary_plot(self.shap_values))

    # def force_plot(self, sample_index):
    #     st_shap(shap.force_plot(self.shap_values[sample_index], matplotlib=True), width=700)

    # def bar_plot(self, sample_index):
    #     st_shap(shap.bar_plot(self.shap_values.values[sample_index]))

    # def waterfall_plot(self, sample_index):
    #     st_shap(shap.plots.waterfall(self.shap_values[sample_index]))

    # def heatmap_plot(self):
    #     st_shap(shap.plots.heatmap(self.shap_values))