from sklearn.metrics import confusion_matrix
from plotting import plot_confusion


# Predict data
def make_confusion_matrix(y_true, y_pred, labels, out_path):

    confusion = confusion_matrix(y_true, y_pred)
    plot_confusion(confusion, labels, out_path)