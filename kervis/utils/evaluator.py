import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


class Evaluator():
    """
    For evaluating the model

    Parameters
    ----------
    model : Model
        The model to be evaluated

    Attributes
    ----------
    model : Model
        The model to be evaluated

    functions
    ---------
    accuracy()
        Return the accuracy of the model

    confusion_matrix()
        Plot the confusion matrix of the model

    roc_curve()
        Plot the ROC curve of the model
    """
    def __init__(self, model):
        self.model = model

    def accuracy(self):
        return accuracy_score(self.model.y_test, self.model.y_pred)
    
    def confusion_matrix(self, path='./plots/result/cm/', cmap='Blues'):
        # change the figure size
        plt.figure(figsize=(4, 2), dpi=300)
        cm = confusion_matrix(self.model.y_test, self.model.y_pred)
        ax = sns.heatmap(cm, 
                    annot=[[f"TP\n{cm[0][0]:.0f}", f"FN\n{cm[0][1]:.0f}"], [f"FP\n{cm[1][0]:.0f}", f"TN\n{cm[1][1]:.0f}"]], fmt='', cmap=cmap, linewidth=.5)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_xticklabels(['Positive', 'Negative'])
        ax.set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
        # remove the color bar on the right side
        ax.collections[0].colorbar.remove()
        plt.title(self.model.name + ' Confusion Matrix')
        plt.savefig('{}{} cm.png'.format(path, self.model.name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def roc_curve(self, plt, color = 'darkorange'):
        fpr, tpr, thresholds = roc_curve(self.model.y_test, self.model.y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=self.model.name +' (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        # plt.show()
