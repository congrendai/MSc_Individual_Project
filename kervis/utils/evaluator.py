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
        # change the figure size
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(self.model.y_test, self.model.y_pred)
        ax = sns.heatmap(cm, 
                    annot=[[f"TP\n{cm[0][0]:.0f}", f"FN\n{cm[0][1]:.0f}"], [f"FP\n{cm[1][0]:.0f}", f"TN\n{cm[1][1]:.0f}"]], fmt='', cmap='Blues', linewidth=.5)
        ax.set_xlabel('Predicted labels')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('True labels')
        # remove xticks and yticks label
        ax.set_xticklabels(['Positive', 'Negative'])
        # set the xticks label to the top
        ax.xaxis.set_ticks_position('top')
        ax.set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
        # remove the color bar on the right side
        ax.collections[0].colorbar.remove()
        plt.show()
    
    def classification_report(self):
        print(classification_report(self.model.y_test, self.model.y_pred))
    
    def roc_curve(self, color = 'darkorange'):
        fpr, tpr, thresholds = roc_curve(self.model.y_test, self.model.y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=self.model.name +' (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        # plt.show()

    
    def precision_recall_curve(self):
        precision, recall, thresholds = precision_recall_curve(self.model.y_test, self.model.y_pred)
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()
