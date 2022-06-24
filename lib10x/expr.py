import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import numpy as np
import scipy.sparse as sp_sparse
import tables
import pandas as pd

from scipy.spatial import distance
import networkx as nx
import os
import phenograph
import libplot
import libcluster
import libtsne
import seaborn as sns
from .constants import EDGE_WIDTH, EDGE_COLOR, MARKER_SIZE, SUBPLOT_SIZE

EXP_ALPHA = 0.8

def base_expr_plot(data,
                   exp,
                   dim=[1, 2],
                   cmap=plt.cm.plasma,
                   marker='o',
                   edgecolors=EDGE_COLOR,
                   linewidth=1,
                   s=MARKER_SIZE,
                   alpha=1,
                   w=libplot.DEFAULT_WIDTH,
                   h=libplot.DEFAULT_HEIGHT,
                   fig=None,
                   ax=None,
                   norm=None):  # plt.cm.plasma):
    """
    Base function for creating an expression plot for T-SNE/2D space
    reduced representation of data.

    Parameters
    ----------
    data : Pandas dataframe
        features x dimensions, e.g. rows are cells and columns are tsne dimensions
    exp : numpy array
        expression values for each data point so it must have the same number
        of elements as data has rows.
    d1 : int, optional
        First dimension being plotted (usually 1)
    d2 : int, optional
        Second dimension being plotted (usually 2)
    fig : matplotlib figure, optional
        Supply a figure object on which to render the plot, otherwise a new
        one is created.
    ax : matplotlib ax, optional
        Supply an axis object on which to render the plot, otherwise a new
        one is created.
    norm : Normalize, optional
        Specify how colors should be normalized

    Returns
    -------
    fig : matplotlib figure
        If fig is a supplied argument, return the supplied figure, otherwise
        a new figure is created and returned.
    ax : matplotlib axis
        If ax is a supplied argument, return this, otherwise create a new
        axis and attach to figure before returning.
    """

    if ax is None:
        fig, ax = libplot.new_fig(w=w, h=h)

    # if norm is None and exp.min() < 0:
    #norm = matplotlib.colors.Normalize(vmin=-3, vmax=3, clip=True)

    if norm is None:
        norm = libplot.NORM_3

    # Sort by expression level so that extreme values always appear on top
    idx = np.argsort(exp) #np.argsort(abs(exp))  # np.argsort(exp)

    print(data.shape, idx)

    x = data.iloc[idx, dim[0] - 1].values  # data['{}-{}'.format(t, d1)][idx]
    y = data.iloc[idx, dim[1] - 1].values  # data['{}-{}'.format(t, d2)][idx]
    e = exp[idx]

    # if (e.min() == 0):
    #print('Data does not appear to be z-scored. Transforming now...')
    # zscore
    #e = (e - e.mean()) / e.std()

    #print(e.min(), e.max())

    # z-score
    #e = (e - e.mean()) / e.std()

    # limit to 3 std for z-scores
    #e[e < -3] = -3
    #e[e > 3] = 3

    ax.scatter(x,
               y,
               c=e,
               s=s,
               marker=marker,
               alpha=alpha,
               cmap=cmap,
               norm=norm,
               edgecolors='none',  # edgecolors,
               linewidth=linewidth)

#    for i in range(0, x.size):
#        en = norm(e[i])
#        color = cmap(int(en * cmap.N))
#        color = np.array(color)
#
#        c1 = color.copy()
#        c1[-1] = 0.5
#
#        #print(c1)
#
#        ax.scatter(x[i],
#               y[i],
#               c=[c1],
#               s=s,
#               marker=marker,
#               edgecolors='none', #edgecolors,
#               linewidth=linewidth)
#
#
#
#        mean = color.mean()
#
#        #print(x[i], y[i], mean)
#
#        #if mean > 0.5:
#        ax.scatter(x[i],
#               y[i],
#               c='#ffffff00',
#               s=s,
#               marker=marker,
#               norm=norm,
#               edgecolors=[color],
#               linewidth=linewidth)

    #libcluster.format_axes(ax, title=t)

    return fig, ax


def expr_plot(data,
              exp,
              dim=[1, 2],
              cmap=plt.cm.magma,
              marker='o',
              s=MARKER_SIZE,
              alpha=1,
              edgecolors=EDGE_COLOR,
              linewidth=EDGE_WIDTH,
              w=libplot.DEFAULT_WIDTH,
              h=libplot.DEFAULT_HEIGHT,
              show_axes=False,
              fig=None,
              ax=None,
              norm=None,
              colorbar=False):  # plt.cm.plasma):
    """
    Creates a base expression plot and adds a color bar.
    """

    is_first = False

    if ax is None:
        fig, ax = libplot.new_fig(w, h)
        is_first = True

    base_expr_plot(data,
                   exp,
                   dim=dim,
                   s=s,
                   marker=marker,
                   edgecolors=edgecolors,
                   linewidth=linewidth,
                   alpha=alpha,
                   cmap=cmap,
                   norm=norm,
                   w=w,
                   h=h,
                   ax=ax)

    # if colorbar or is_first:
    if colorbar:
        libplot.add_colorbar(fig, cmap, norm=norm)
        #libcluster.format_simple_axes(ax, title=t)

    if not show_axes:
        libplot.invisible_axes(ax)

    return fig, ax


# def expr_plot(tsne,
#                   exp,
#                   d1=1,
#                   d2=2,
#                   x1=None,
#                   x2=None,
#                   cmap=BLUE_YELLOW_CMAP,
#                   marker='o',
#                   s=MARKER_SIZE,
#                   alpha=EXP_ALPHA,
#                   out=None,
#                   fig=None,
#                   ax=None,
#                   norm=None,
#                   w=libplot.DEFAULT_WIDTH,
#                   h=libplot.DEFAULT_HEIGHT,
#                   colorbar=True): #plt.cm.plasma):
#    """
#    Creates a basic t-sne expression plot.
#
#    Parameters
#    ----------
#    data : pandas.DataFrame
#        t-sne 2D data
#    """
#
#    fig, ax = expr_plot(tsne,
#                        exp,
#                        t='TSNE',
#                        d1=d1,
#                        d2=d2,
#                        x1=x1,
#                        x2=x2,
#                        cmap=cmap,
#                        marker=marker,
#                        s=s,
#                        alpha=alpha,
#                        fig=fig,
#                        ax=ax,
#                        norm=norm,
#                        w=w,
#                        h=h,
#                        colorbar=colorbar)
#
#    set_tsne_ax_lim(tsne, ax)
#
#    libplot.invisible_axes(ax)
#
#    if out is not None:
#        libplot.savefig(fig, out, pad=0)
#
#    return fig, ax


def create_expr_plot(tsne,
                     exp,
                     dim=[1, 2],
                     cmap=None,
                     marker='o',
                     s=MARKER_SIZE,
                     alpha=EXP_ALPHA,
                     fig=None,
                     ax=None,
                     w=libplot.DEFAULT_WIDTH,
                     h=libplot.DEFAULT_HEIGHT,
                     edgecolors=EDGE_COLOR,
                     linewidth=EDGE_WIDTH,
                     norm=None,
                     method='tsne',
                     show_axes=False,
                     colorbar=True,
                     out=None):  # plt.cm.plasma):
    """
    Creates and saves a presentation tsne plot
    """

    if out is None:
        out = '{}_expr.pdf'.format(method)

    fig, ax = expr_plot(tsne,
                        exp,
                        dim=dim,
                        cmap=cmap,
                        marker=marker,
                        s=s,
                        alpha=alpha,
                        fig=fig,
                        w=w,
                        h=h,
                        ax=ax,
                        show_axes=show_axes,
                        colorbar=colorbar,
                        norm=norm,
                        linewidth=linewidth,
                        edgecolors=edgecolors)

    if out is not None:
        libplot.savefig(fig, out, pad=0)

    return fig, ax


def base_pca_expr_plot(data,
                       exp,
                       dim=[1, 2],
                       cmap=None,
                       marker='o',
                       s=MARKER_SIZE,
                       alpha=EXP_ALPHA,
                       fig=None,
                       ax=None,
                       norm=None):  # plt.cm.plasma):
    fig, ax = base_expr_plot(data,
                             exp,
                             t='PC',
                             dim=dim,
                             cmap=cmap,
                             marker=marker,
                             s=s,
                             fig=fig,
                             alpha=alpha,
                             ax=ax,
                             norm=norm)

    return fig, ax


def pca_expr_plot(data,
                  expr,
                  name,
                  dim=[1, 2],
                  cmap=None,
                  marker='o',
                  s=MARKER_SIZE,
                  alpha=EXP_ALPHA,
                  fig=None,
                  ax=None,
                  norm=None):  # plt.cm.plasma):
    out = 'pca_expr_{}_t{}_vs_t{}.pdf'.format(name, 1, 2)

    fig, ax = base_pca_expr_plot(data,
                                 expr,
                                 dim=dim,
                                 cmap=cmap,
                                 marker=marker,
                                 s=s,
                                 alpha=alpha,
                                 fig=fig,
                                 ax=ax,
                                 norm=norm)

    libplot.savefig(fig, out)
    plt.close(fig)

    return fig, ax


def expr_grid_size(x, size=SUBPLOT_SIZE):
    """
    Auto size grid to look nice.
    """

    if type(x) is int:
        l = x
    elif type(x) is list:
        l = len(x)
    elif type(x) is np.ndarray:
        l = x.shape[0]
    elif type(x) is pd.core.frame.DataFrame:
        l = x.shape[0]
    else:
        return None

    cols = int(np.ceil(np.sqrt(l)))

    w = size * cols

    rows = int(l / cols) + 2

    if l % cols == 0:
        # Assume we will add a row for a color bar
        rows += 1

    h = size * rows

    return w, h, rows, cols
