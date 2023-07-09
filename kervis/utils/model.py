import shap
import scipy
from sklearn.svm import SVC
from kervis.utils.dataset import Dataset
from sklearn.metrics import accuracy_score
from kervis.kernels import ShortestPath, Graphlet
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, dataset_name, kernel , model, test_size=0.2, shuffle=False):
        self.kernel = kernel()
        self.dataset = Dataset(dataset_name)
        if type(self.kernel) == type(ShortestPath()) or type(self.kernel) == type(Graphlet()):
            self.kernel.fit_transform(self.dataset.graphs)
        else:
            self.kernel.fit_transform(self.dataset.data)    
        
        self.features = self.kernel.X
        
        if type(self.features) == scipy.sparse.csr.csr_matrix:
            self.features= self.features.toarray()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.dataset.y, test_size=test_size, shuffle=shuffle)

        if model == 'SVM':
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