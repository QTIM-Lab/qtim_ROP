import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

CLASSES = ['No', 'Pre-Plus', 'Plus']


def plot_accuracy(history, out_file=None):

    plt.figure()
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig(out_file)


def plot_loss(history, out_file=None):

    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig(out_file)


def plot_confusion(confusion, classes, out_file):

    fig, ax = plt.subplots()
    df_cm = pd.DataFrame(confusion, index=classes, columns=classes)

    df_cm = df_cm.reindex_axis(CLASSES, axis=0)
    df_cm = df_cm.reindex_axis(CLASSES, axis=1)

    sns.heatmap(df_cm, cmap='Blues', annot=True, fmt='g')
    ax.xaxis.tick_top()
    sns.plt.savefig(out_file)
