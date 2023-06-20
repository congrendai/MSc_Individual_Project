import shap
from sklearn.svm import SVC
from streamlit_shap import st_shap
from utils import fetch_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, kernel, dataset_name):
        self.kernel = kernel(normalize=False)
        self.dataset = fetch_dataset(dataset_name, verbose=False)
        self.readme = self.dataset.readme
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data, self.dataset.target, test_size=0.2, shuffle=False)
        self.K_train = self.kernel.fit_transform(self.X_train)
        self.K_test = self.kernel.transform(self.X_test)
        self.clf = SVC(kernel='precomputed')
        self.clf.fit(self.K_train, self.y_train)
        self.y_pred = self.clf.predict(self.K_test)
        self.explainer = shap.Explainer(self.clf.predict, self.K_train, max_evals=2*len(self.K_train)+1)
        self.shap_values = self.explainer(self.K_test)
        print("Accuracy for {} is {}".format(dataset_name, accuracy_score(self.y_test, self.y_pred)))

    def get_readme(self):
        return self.readme

    def summary_plot(self):
        st_shap(shap.summary_plot(self.shap_values))

    def force_plot(self, sample_index):
        st_shap(shap.force_plot(self.shap_values[sample_index], matplotlib=True), width=700)

    def bar_plot(self, sample_index):
        st_shap(shap.bar_plot(self.shap_values.values[sample_index]))

    def waterfall_plot(self, sample_index):
        st_shap(shap.plots.waterfall(self.shap_values[sample_index]))

    def heatmap_plot(self):
        st_shap(shap.plots.heatmap(self.shap_values))