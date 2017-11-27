import matplotlib.pyplot as plt
import numpy as np

import itertools

PLT_DPI = 100

def show_image_grid(images, rows, cols, height, width, title, subtitles=None):
    amin = np.amin(images)
    if amin < 0:
        images += -1 * amin
    amax = np.amax(images)
    if amax > 1:
        images = images / amax
    plt.close()
    fig, ax_arr = plt.subplots(rows, cols)
    fig.set_dpi(PLT_DPI)
    fig.set_size_inches(width, height)
    plt.suptitle(title, fontsize=22)
    plt.tight_layout(rect=(0, 0, 1, 0.85))
    i = 0
    for x in range(rows):
        for y in range(cols):
            ax_arr[x*cols + y].imshow(images[i, ...])
            if subtitles is not None:
                ax_arr[x*cols + y].set_title(subtitles[i], fontsize=16)
            i += 1
    plt.show()
    plt.close()

def plot_confusion_matrix(mtx, title, height, width):
    """Plot a confusion matrix.

    Args:
        mtx (np.ndarray): An output of metrics.ConfusionMatrixMetric.evaluate.
        title (str): A title for the plot.
        height (int): The height of the plot, in inches.
        width (int): The width of the plot, in inches.
    """
    num_classes = mtx.shape[0]
    plt.close()
    plt.figure(figsize=(width, height), dpi=PLT_DPI)
    plt.imshow(np.transpose(mtx), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    thresh = mtx.max() / 2
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        plt.text(j, i, format(mtx[j, i], 'd'),
                 horizontalalignment="center",
                 color="white" if mtx[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()
    plt.close()

def plot_line(xs, ys, title, x_label, y_label, height, width):
    """Plot a line.

    Args:
        xs (list of float): The x-values.
        ys (list of float): The y-values.
        title (str): The title.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        height (int): The height of the plot, in inches.
        width (int): The width of the plot, in inches.
    """
    plt.close()
    plt.figure(figsize=(width, height), dpi=PLT_DPI)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(xs, ys)
    plt.show()
    plt.close()

def plot_lines(xs, sets_of_ys, title, x_label, y_label, line_labels, height,
        width):
    """Plot some lines.

    Args:
        xs (list of float): The x-values.
        sets_of_ys (list of np.ndarray): The y-values. The y-value of the j-th
            line at xs[i] is ys[i][j].
        title (str): The title.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        line_labels (list of str): The labels for the lines.
        height (int): The height of the plot, in inches.
        width (int): The width of the plot, in inches.
    """
    plt.close()
    fig = plt.figure(figsize=(width, height), dpi=PLT_DPI)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i, line_label in enumerate(line_labels):
        ys = [set_of_ys[i] for set_of_ys in sets_of_ys]
        if i < 7:
            color = (i/7, 0, 0)
        elif i >= 7:
            color = (0, (i-7)/7, 0)
        plt.plot(xs, ys, label=line_label, color=color, linewidth=0.5)
    plt.show()
    plt.close()