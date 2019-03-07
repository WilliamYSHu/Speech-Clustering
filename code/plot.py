import pylab
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram,set_link_color_palette,linkage,fcluster,leaders
from scipy.spatial.distance import pdist,squareform

def plot_embedding_annotation(emb, digits, threshold, ax=None, rescale=True):
    """
    This function is used to visualize the learned low-d features
    We intend to see if we learn to disentangle factors of variations
    @emb : the input low-d feature
    @digits : the immage annotation of emb
    @threshold: minimal distances between two points
    """

    # Rescaling
    if rescale:
        x_min, x_max = np.min(emb, 0), np.max(emb, 0)
        emb = (emb - x_min) / (x_max - x_min)

    _, ax = plt.subplots()

    if hasattr(offsetbox, 'AnnotationBbox'):
        mycanvas = np.array([[1., 1.]])
        for i in range(digits.shape[0]):
            dist = np.sum((emb[i] - mycanvas) ** 2, 1)
            if np.min(dist) < threshold:
                # don't show points that are too close
                # You may try different threshold
                continue
            mycanvas = np.r_[mycanvas, [emb[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits[i], cmap=plt.cm.gray_r),
                emb[i],
                frameon=False)
            ax.add_artist(imagebox)
    ax.set_xticks([])
    ax.set_yticks([])
    return 0


def plot_embedding(emb, labels, ax=None, rescale=False):
    """
    This function is used to visualize the learned low-d features
    We intend to see cluster information via visualization
    @emb : the input low-d feature
    @label : the text annotation of emb
    """
    # Rescaling
    if rescale:
        x_min, x_max = np.min(emb, 0), np.max(emb, 0)
        emb = (emb - x_min) / (x_max - x_min)
    _, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    for i, _ in enumerate(colors):
        ax.scatter(emb[labels == i, 0], emb[labels == i, 1], c=colors[i], label=i, edgecolors='k')
    ax.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    return 0


def plot_images(X, nrow, ncolumn, rowscale, columnscale):
    """
    This function is used to plot images of the digits
    @X : the input images
    @nrow : number of images per row in canvas
    @ncolumn: number of images per column in canvas
    @rowscale,@columnscale: image scale
    """

    _, ax = plt.subplots()
    imgcanvas = np.zeros(((rowscale+2) * nrow, (columnscale+2) * ncolumn))
    for i in range(nrow):
        ix = (rowscale+2) * i + 1
        for j in range(ncolumn):
            iy = (columnscale+2) * j + 1
            imgcanvas[ix:ix + rowscale, iy:iy + columnscale] = X[i * ncolumn + j]

    ax.imshow(imgcanvas, cmap=plt.cm.binary)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
   
def plot_confusion_matrix(cm, classes, figname=None, normalize=False, cmap=plt.cm.Blues, rotation=45, figsize=None):
    """
    This function plots the confusion matrix.
    @cm: confusion matrix
    @classes: class names
    @normalize: if True normalize each row
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalize confusion matrix per row")

    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=rotation)
    ax.set_yticklabels(classes)

    fmt = '%.2f' if normalize else '%d'
    thresh = cm.max() / 2.

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)
    return

def plot_dendrogram(linkmap, labels, figname=None, ratio=0.6, orientation='top', figsize=(15, 10), rotation=-45.0, fontsize=6.5):
    """
    This function is used to plot dendrogram, using scipy-dendrogram
    @linkmap - linkage distance of feature e.g. (linkage(dist(X)))
    @labels - labels of feature (X)
    @ratio - scalar to control 'color_threshold'
    @figname - save your figure as figname
    """
    plt.figure(figsize=figsize)

    Z = dendrogram(linkmap,
                    color_threshold=ratio*max(linkmap[:,2]),
                    labels=labels,
                    show_leaf_counts=True,
                    leaf_rotation=rotation,
                    leaf_font_size=fontsize,
                    show_contracted=True,
                    orientation=orientation)
    if orientation in {'top', 'bottom'}:
        plt.yticks([])
    else:
        plt.xticks([])
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, dpi=100)
    return

def plot_heat_dendrogram(Y1, Y2, dist, labels, figname, cmap=pylab.cm.YlGnBu, ratio=0.6):

    """
    This function allows you to compare two clustering method, e.g. centroid vs single,
    @feature is your input feature [nsample, ndim]
    @title is the name of your plot
    @method1/method2, two methods for comparison
    @cmap, color map to use
    """

    Dist_Matrix = squareform(dist)

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(25,25))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Z1 = dendrogram(Y1, orientation='right', color_threshold = ratio*max(Y1[:,2]))
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Z2 = dendrogram(Y2, color_threshold = ratio*max(Y2[:,2]))
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    Dist_Matrix = Dist_Matrix[idx1,:]
    Dist_Matrix = Dist_Matrix[:,idx2]
    idx1 = labels[Z1['leaves']]
    idx2 = labels[Z2['leaves']]
    im = axmatrix.matshow(Dist_Matrix, aspect='auto', origin='lower', cmap=cmap)

    axmatrix.set_xticks(range(np.shape(Dist_Matrix)[0]))
    axmatrix.set_xticklabels(idx1, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=9)

    axmatrix.set_yticks(range(np.shape(Dist_Matrix)[0]))
    axmatrix.set_yticklabels(idx2, minor=False)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()

    # Plot colorbar
    axcolor = fig.add_axes([0.95,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    if not figname:
        fig.savefig(figname)
