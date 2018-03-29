import itertools, math

import matplotlib.pyplot as plt
import numpy as np

PLT_DPI = 300

def plot_images(images, cols, dims, title, subtitles=None, path=None):
    """Plot images in a grid.

    Args:
        images (np.ndarray): The images. Must have shape (num_images, ...).
        cols (int): The number of grid columns.
        dims (tuple of float): The dimensions to plot each image with, as a tuple of height and width, both in inches.
        title (str): The title for the plot.
        subtitles (list of str): The list of subtitles for the images. The i-th element is the subtitle for the i-th
            image. If omitted or None, no subtitles are plotted.
        path (str): The path to save the plot to. If ``None`` or omitted, the
            plot is shown.
    """
    TITLE_FONT_SIZE = 20
    SUBTITLE_FONT_SIZE = 12
    TOP_SUBPLOTS_ADJUST_FIXED = 0.9
    TIGHT_LAYOUT_H_PAD = 2.5
    TIGHT_LAYOUT_W_PAD = 1

    assert cols > 0, "argument cols must be > 0"
    assert len(dims) == 2, "argument dims must have length 2"
    assert subtitles is None or len(subtitles) == images.shape[0], "argument subtitles must be omitted, None, or a " \
                                                                   "sequence of length images.shape[0]"

    images_min = np.amin(images)
    if images_min < 0:
        images += -1 * images_min
    images_max = np.amax(images)
    if images_max > 1:
        images = images / images_max

    plt.figure(dpi=PLT_DPI)

    rows = math.ceil(images.shape[0] / cols)
    fig, ax_arr = plt.subplots(rows, cols)
    if images.shape[0] > 1:
        ax_arr = ax_arr.flatten()
    else:
        ax_arr = np.array([ax_arr])
    fig_width, fig_height = cols*dims[1], rows*dims[1]
    fig.set_size_inches(fig_width, fig_height)
    fig.tight_layout(h_pad=TIGHT_LAYOUT_H_PAD, w_pad=TIGHT_LAYOUT_W_PAD)
    fig.subplots_adjust(top=fig_height/(fig_height+TOP_SUBPLOTS_ADJUST_FIXED))

    for i in range(images.shape[0]):
        ax_arr[i].imshow(images[i, ...])
        if subtitles is not None:
            ax_arr[i].set_title(subtitles[i], fontsize=SUBTITLE_FONT_SIZE)

    for i in range(images.shape[0], ax_arr.shape[0]):
        ax_arr[i].set_axis_off()

    plt.suptitle(title, fontsize=TITLE_FONT_SIZE)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi='figure', format='svg')
    plt.close()

def plot_confusion_matrix(mtx, title, height, width, path=None):
    """Plot a confusion matrix.

    Args:
        mtx (np.ndarray): An output of metrics.ConfusionMatrixMetric.evaluate.
        title (str): A title for the plot.
        height (int): The height of the plot, in inches.
        width (int): The width of the plot, in inches.
        path (str): The path to save the plot to. If ``None`` or omitted, the
            plot is shown.
    """
    num_classes = mtx.shape[0]
    plt.close()
    plt.figure(figsize=(width, height), dpi=PLT_DPI)
    plt.imshow(mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    thresh = mtx.max() / 2
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        plt.text(i, j, format(mtx[j, i], 'd'),
                 horizontalalignment="center",
                 color="white" if mtx[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi='figure', format='svg')
    plt.close()

def plot_line(xs, ys, title, x_label, y_label, height, width, path=None):
    """Plot a line.

    Args:
        xs (list of float): The x-values.
        ys (list of float): The y-values.
        title (str): The title.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        height (int): The height of the plot, in inches.
        width (int): The width of the plot, in inches.
        path (str): The path to save the plot to. If ``None`` or omitted, the
            plot is shown.
    """
    plt.close()
    plt.figure(figsize=(width, height), dpi=PLT_DPI)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(xs, ys)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi='figure', format='svg')
    plt.close()

def plot_lines(xs, sets_of_ys, title, x_label, y_label, line_labels, height,
        width, line_styles=None, path=None):
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
        line_styles (list(str)): Styles for the lines. If ``None`` or omitted,
            styles are chosen automatically. Each must be a valid format string
            for `matplotlib.pyplot.plot`.
        path (str): The path to save the plot to. If ``None`` or omitted, the
            plot is shown.
    """
    if line_styles is None:
        line_styles = ['' for _ in range(len(sets_of_ys[0]))]

    plt.close()
    plt.figure(figsize=(width, height), dpi=PLT_DPI)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in range(len(sets_of_ys[0])):
        ys = [set_of_ys[i] for set_of_ys in sets_of_ys]
        line_style, line_label = line_styles[i], line_labels[i]
        plt.plot(xs, ys, line_style, label=line_label)
    plt.legend()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi='figure', format='svg')
    plt.close()


def plot_scatter(xs, ys, title, x_label, y_label, height, width, colors=None,
                 path=None):
    """Make a scatterplot.

    Args:
        xs (list of float): The x-values.
        ys (list of float): The y-values.
        title (str): The title.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        height (int): The height of the plot, in inches.
        width (int): The width of the plot, in inches.
        colors (list of float): The colors. If omitted or ``None``, all black.
        path (str): The path to save the plot to. If ``None`` or omitted, the
            plot is shown.
    """
    if colors is None:
        colors = [1 for _ in xs]
    plt.close()
    plt.figure(figsize=(width, height), dpi=PLT_DPI)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(xs, ys, c=colors)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi='figure', format='svg')
    plt.close()
