from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from qtim_ROP.utils.common import series_to_plot_dict


CLASSES = ['No', 'Pre-Plus', 'Plus']


def plot_accuracy(history, out_file=None):

    plt.figure()
    plt.plot(history['acc'])

    legend = ['Training']
    if history.get('val_acc'):
        plt.plot(history['val_acc'])
        legend.append('Validation')

    plt.title('Model accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(out_file)


def plot_loss(history, out_file=None):

    plt.figure()
    plt.plot(history['loss'])

    legend = ['Training']
    if history.get('val_loss'):
        plt.plot(history['val_loss'])
        legend.append('Validation')

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(out_file)


def plot_LR(lr, out_file):

    plt.figure()
    plt.plot(lr)
    plt.savefig(out_file)


def plot_confusion(confusion, classes, out_file):

    fig, ax = plt.subplots()
    df_cm = pd.DataFrame(confusion, index=classes, columns=classes)

    if len(classes) > 2:
        df_cm = df_cm.reindex_axis(CLASSES, axis=0)
        df_cm = df_cm.reindex_axis(CLASSES, axis=1)

    sns.heatmap(df_cm, cmap='Blues', annot=True, fmt='g', annot_kws={"size": 16})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    sns.plt.xlabel('Automated diagnosis')
    sns.plt.ylabel('Reference standard diagnosis')
    sns.plt.savefig(out_file)


def plot_counts(count_series, x, y, order, y_label, title, out_path):

    df = series_to_plot_dict(count_series, x, y)
    fig, ax = sns.plt.subplots()
    sns.barplot(data=df, x=x, y=y, order=order)
    sns.plt.ylabel(y_label)
    sns.plt.title(title)
    sns.plt.savefig(out_path)


