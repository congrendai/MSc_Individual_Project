import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve


class Evaluator():
    def __init__(self, model):
        self.model = model

    def accuracy(self):
        return accuracy_score(self.model.y_test, self.model.y_pred)
    
    def precision(self):
        return precision_score(self.model.y_test, self.model.y_pred)
    
    def recall(self):
        return recall_score(self.model.y_test, self.model.y_pred)
    
    def f1(self):
        return f1_score(self.model.y_test, self.model.y_pred)
    
    def confusion_matrix(self):
        ax = sns.heatmap(confusion_matrix(self.model.y_test, self.model.y_pred), 
                    annot=True, cmap='Blues', linewidth=.5)
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    
    def classification_report(self):
        print(classification_report(self.model.y_test, self.model.y_pred))
    
    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.model.y_test, self.model.y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    
    def precision_recall_curve(self):
        precision, recall, thresholds = precision_recall_curve(self.model.y_test, self.model.y_pred)
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()