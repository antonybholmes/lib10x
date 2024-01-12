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
import libtsne
import seaborn as sns
from .constants import *
from .lib10x import get_gene_names, get_gene_ids, mkdir
from .color import *
from libsparse import SparseDataFrame
from PIL import Image, ImageFilter

EXPR_ALPHA = 0.8

def get_gene_data(data, g, ids=None, gene_names=None):
    if ids is None:
        ids, gene_names = get_gene_names(data)

    if isinstance(g, list):
        g = np.array(g)

    if isinstance(g, np.ndarray):
        idx = np.where(np.isin(ids, g))[0]

        if idx.size < 1:
            # if id does not exist, try the gene names
            idx = np.where(np.isin(gene_names, g))[0]

            if idx.size < 1:
                return None
    else:
        idx = np.where(ids == g)[0]

        if idx.size > 0:
            # if id exists, pick the first
            idx = idx[0]
        else:
            # if id does not exist, try the gene names
            idx = np.where(gene_names == g)[0]

            if idx.size > 0:
                idx = idx[0]
            else:
                return None

    if isinstance(data, SparseDataFrame):
        return data[idx, :].to_array()
    else:
        return data.iloc[idx, :].values


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
    idx = np.argsort(exp)   # np.argsort(abs(exp))  # np.argsort(exp)

    print(data.shape, idx, dim)

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

    #format_axes(ax, title=t)

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
        #format_simple_axes(ax, title=t)

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
#                   alpha=EXPR_ALPHA,
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
                     alpha=EXPR_ALPHA,
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
                       alpha=EXPR_ALPHA,
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
                  alpha=EXPR_ALPHA,
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


def gene_expr_grid(data, tsne, genes, cmap=None, size=SUBPLOT_SIZE):
    """
    Plot multiple genes on a grid.

    Parameters
    ----------
    data : Pandas dataframe
        Genes x samples expression matrix 
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    genes : array
        List of gene names

    Returns
    -------
    fig : Matplotlib figure
        A new Matplotlib figure used to make the plot
    """

    if type(genes) is pd.core.frame.DataFrame:
        genes = genes['Genes'].values

    ids, gene_names = get_gene_names(data)

    gene_ids = get_gene_ids(data, genes, ids=ids, gene_names=gene_names)

    w, h, rows, cols = expr_grid_size(gene_ids, size=size)

    fig = libplot.new_base_fig(w=w, h=h)

    for i in range(0, len(gene_ids)):
        # gene id
        gene_id = gene_ids[i][1]
        gene = gene_ids[i][2]

        print(gene, gene_id)

        exp = get_gene_data(data, gene_id, ids=ids, gene_names=gene_names)

        ax = libplot.new_ax(fig, rows, cols, i + 1)

        expr_plot(tsne, exp, ax=ax, cmap=cmap, colorbar=False)

        # if i == 0:
        #    format_axes(ax)
        # else:

        # libplot.invisible_axes(ax)

        ax.set_title('{} ({})'.format(gene_ids[i][2], gene_ids[i][1]))

    libplot.add_colorbar(fig, cmap)

    return fig


def genes_expr(data,
               tsne,
               genes,
               prefix='',
               dim=[1, 2],
               index=None,
               dir='GeneExp',
               cmap=BGY_CMAP,
               norm=None,
               w=4,
               h=4,
               s=30,
               alpha=ALPHA,
               linewidth=EDGE_WIDTH,
               edgecolors='none',
               colorbar=True,
               outline=True,
               method='tsne',
               format='png'):
    """
    Plot multiple genes on a grid.

    Parameters
    ----------
    data : Pandas dataframe
        Genes x samples expression matrix 
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    genes : array
        List of gene names
    """

    if dir[-1] == '/':
        dir = dir[:-1]

    if not os.path.exists(dir):
        mkdir(dir)

    if index is None:
        index = data.index

    if isinstance(genes, pd.core.frame.DataFrame):
        genes = genes['Genes'].values

    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=-3, vmax=3, clip=True)

    #cmap = plt.cm.plasma

    ids, gene_names = get_gene_names(data)

    print(ids, gene_names, genes)

    gene_ids = get_gene_ids(data, genes, ids=ids, gene_names=gene_names)

    print(gene_ids)

    for i in range(0, len(gene_ids)):
        gene_id = gene_ids[i][1]
        gene = gene_ids[i][2]

        print(gene_id, gene)

        exp = get_gene_data(data, gene_id, ids=ids, gene_names=gene_names)

        #fig, ax = libplot.new_fig()

        #expr_plot(tsne, exp, ax=ax)

        #libplot.add_colorbar(fig, cmap)

        fig, ax = expr_plot(tsne,
                            exp,
                            cmap=cmap,
                            dim=dim,
                            w=w,
                            h=h,
                            s=s,
                            colorbar=colorbar,
                            norm=norm,
                            alpha=alpha,
                            linewidth=linewidth,
                            edgecolors=edgecolors)

        if gene_id != gene:
            out = '{}/{}_expr_{}_{}.{}'.format(dir,
                                               method, gene, gene_id, format)
        else:
            out = '{}/{}_expr_{}.{}'.format(dir, method, gene, format)

        libplot.savefig(fig, out, pad=0)

        if outline:
            libplot.savefig(fig, 'tmp.png', pad=0)
            
            

            im1 = Image.open('tmp.png')

            # Edge detect on what is left (the clusters)
            imageWithEdges = im1.filter(ImageFilter.FIND_EDGES)
            im_data = np.array(imageWithEdges.convert('RGBA'))

            #r = data[:, :, 0]
            #g = data[:, :, 1]
            #b = data[:, :, 2]
            a = im_data[:, :, 3]

            # (r < 255) | (g < 255) | (b < 255) #(r > 0) & (r == g) & (r == b) & (g == b)
            black_areas = (a > 0)

            d = im_data[np.where(black_areas)]
            d[:, 0:3] = [64, 64, 64]
            im_data[np.where(black_areas)] = d

            im2 = Image.fromarray(im_data)
            im2.save('edges.png', 'png')

            # overlay edges on top of original image to highlight cluster
            # enable if edges desired
            im1.paste(im2, (0, 0), im2)
            im1.save(out, 'png')
    
        plt.close(fig)

def genes_expr_outline(data,
                       tsne,
                       genes,
                       prefix='',
                       index=None,
                       dir='GeneExp',
                       cmap=BGY_CMAP,
                       norm=None,
                       w=6,
                       s=30,
                       alpha=1,
                       linewidth=EDGE_WIDTH,
                       edgecolors='none',
                       colorbar=True,
                       method='tsne',
                       bins=10,
                       background=BACKGROUND_SAMPLE_COLOR):
    """
    Plot multiple genes on a grid.

    Parameters
    ----------
    data : Pandas dataframe
        Genes x samples expression matrix 
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    genes : array
        List of gene names
    """

    if dir[-1] == '/':
        dir = dir[:-1]

    if not os.path.exists(dir):
        mkdir(dir)

    if index is None:
        index = data.index

    if isinstance(genes, pd.core.frame.DataFrame):
        genes = genes['Genes'].values

    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=-3, vmax=3, clip=True)

    #cmap = plt.cm.plasma

    ids, gene_names = get_gene_names(data)

    gene_ids = get_gene_ids(data, genes, ids=ids, gene_names=gene_names)

    for i in range(0, len(gene_ids)):
        gene_id = gene_ids[i][1]
        gene = gene_ids[i][2]

        print(gene_id, gene)

        exp = get_gene_data(data, gene_id, ids=ids, gene_names=gene_names)

        bin_means, bin_edges, binnumber = binned_statistic(exp, exp, bins=bins)

        print(binnumber.min(), binnumber.max())

        iw = w * 300
        im_base = imagelib.new(iw, iw)

        for bin in range(0, bins):
            bi = bin + 1
            idx_bin = np.where(binnumber == bi)[0]
            idx_other = np.where(binnumber != bi)[0]

            tsne_other = tsne.iloc[idx_other, :]

            fig, ax = libplot.new_fig(w, w)

            x = tsne_other.iloc[:, 0]
            y = tsne_other.iloc[:, 1]

            libplot.scatter(x,
                            y,
                            c=[background],
                            ax=ax,
                            edgecolors='none',  # bgedgecolor,
                            linewidth=linewidth,
                            s=s)

            #fig, ax = libplot.new_fig()

            #expr_plot(tsne, exp, ax=ax)

            #libplot.add_colorbar(fig, cmap)

            exp_bin = exp[idx_bin]
            tsne_bin = tsne.iloc[idx_bin, :]

            expr_plot(tsne_bin,
                      exp_bin,
                      cmap=cmap,
                      s=s,
                      colorbar=colorbar,
                      norm=norm,
                      alpha=alpha,
                      linewidth=linewidth,
                      edgecolors=edgecolors,
                      ax=ax)

            tmp = 'tmp{}.png'.format(bin)

            libplot.savefig(fig, tmp, pad=0)
            plt.close(fig)

            im = imagelib.open(tmp)
            im_no_bg = imagelib.remove_background(im)
            im_edges = imagelib.edges(im_no_bg)
            im_smooth = imagelib.smooth(im_edges)
            im_outline = imagelib.paste(im_no_bg, im_smooth)
            imagelib.paste(im_base, im_outline, inplace=True)

#             # find gray areas and mask
#            im_data = np.array(im1.convert('RGBA'))
#
#            r = im_data[:, :, 0]
#            g = im_data[:, :, 1]
#            b = im_data[:, :, 2]
#
#            print(tmp, r.shape)
#
#            grey_areas = (r < 255) & (r > 200) & (g < 255) & (g > 200) & (b < 255) & (b > 200)
#
#
#            d = im_data[np.where(grey_areas)]
#            d[:, :] = [255, 255, 255, 0]
#            im_data[np.where(grey_areas)] = d
#
#
#            #edges1 = feature.canny(rgb2gray(im_data))
#
#            #print(edges1.shape)
#
#            #skimage.io.imsave('tmp_canny_{}.png'.format(bin), edges1)
#
#            im2 = Image.fromarray(im_data)
#
#            im_no_gray, im_smooth = smooth_edges(im1, im1)
#
#            # Edge detect on what is left (the clusters)
#            im_edges = im2.filter(ImageFilter.FIND_EDGES)
#
#
#            im_data = np.array(im_edges.convert('RGBA'))
#
#            #r = data[:, :, 0]
#            #g = data[:, :, 1]
#            #b = data[:, :, 2]
#            #a = im_data[:, :, 3]
#
#            # Non transparent areas are edges
#            #black_areas = (a > 0) #(r < 255) | (g < 255) | (b < 255) #(r > 0) & (r == g) & (r == b) & (g == b)
#
#            #d = im_data[np.where(black_areas)]
#            #d[:, 0:3] = [64, 64, 64]
#            #im_data[np.where(black_areas)] = d
#
#            #im3 = Image.fromarray(im_data)
#            #im2.save('edges.png', 'png')
#
#            im_smooth = im_edges.filter(ImageFilter.SMOOTH)
#            im_smooth.save('edges.png', 'png')
#
#            im2.paste(im_smooth, (0, 0), im_smooth)
#
#            im_base.paste(im2, (0, 0), im2)

        if gene_id != gene:
            out = '{}/{}_expr_{}_{}.png'.format(dir, method, gene, gene_id)
        else:
            out = '{}/{}_expr_{}.png'.format(dir, method, gene)

        print(out)

        # overlay edges on top of original image to highlight cluster
        #im_base.paste(im2, (0, 0), im2)
        imagelib.save(im_base, out)


def avg_expr(data,
             tsne,
             genes,
             cid,
             clusters,
             prefix='',
             index=None,
             dir='GeneExp',
             cmap=OR_RED_CMAP,  # BGY_CMAP,
             norm=None,
             w=libplot.DEFAULT_WIDTH,
             h=libplot.DEFAULT_HEIGHT,
             alpha=1.0,
             colorbar=False,
             method='tsne',
             fig=None,
             ax=None,
             sdmax=0.5):
    """
    Plot multiple genes on a grid.

    Parameters
    ----------
    data : Pandas dataframe
        Genes x samples expression matrix 
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    genes : array
        List of gene names
    """

    if dir[-1] == '/':
        dir = dir[:-1]

    if not os.path.exists(dir):
        mkdir(dir)

    if index is None:
        index = data.index

    if isinstance(genes, pd.core.frame.DataFrame):
        genes = genes['Genes'].values

    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=-3, vmax=3, clip=True)

    #cmap = plt.cm.plasma

    ids, gene_names = get_gene_names(data)

    exp = get_gene_data(data, genes, ids=ids, gene_names=gene_names)

    avg = exp.mean(axis=0)

    avg = (avg - avg.mean()) / avg.std()
    avg[avg < -1.5] = -1.5
    avg[avg > 1.5] = 1.5
    avg = (avg - avg.min()) / (avg.max() - avg.min())  # min_max_scale(avg)

    create_expr_plot(tsne,
                     avg,
                     cmap=cmap,
                     w=w,
                     h=h,
                     colorbar=colorbar,
                     norm=norm,
                     alpha=alpha,
                     fig=fig,
                     ax=ax)

    x = tsne.iloc[:, 0].values  # data['{}-{}'.format(t, d1)][idx]
    y = tsne.iloc[:, 1].values  # data['{}-{}'.format(t, d2)][idx]

    idx = np.where(clusters['Cluster'] == cid)[0]

    nx = 500
    ny = 500
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)

    x = x[idx]
    y = y[idx]

    #centroid = [x.sum() / x.size, y.sum() / y.size]
    centroid = [(x * avg[idx]).sum() / avg[idx].sum(),
                (y * avg[idx]).sum() / avg[idx].sum()]

    d = np.array([distance.euclidean(centroid, (a, b)) for a, b in zip(x, y)])

    sd = d.std()
    m = d.mean()

    print(m, sd)

    z = (d - m) / sd

    # find all points within 1 sd of centroid

    idx = np.where(abs(z) < sdmax)[0]  # (d > x1) & (d < x2))[0]
    x = x[idx]
    y = y[idx]

    points = np.array([[p1, p2] for p1, p2 in zip(x, y)])
    hull = ConvexHull(points)

    #x1 = x[idx]
    #y1 = y[idx]
    # avg1 = np.zeros(x.size) #avg[idx]
    #avg1[idx] = 1

#    fx = interp1d(points[hull.vertices, 0], points[hull.vertices, 1], kind='cubic')
#    fy = interp1d(points[hull.vertices, 1], points[hull.vertices, 0], kind='cubic')
#
#    xt = np.linspace(x.min(), x.max(), 100, endpoint=True)
#    yt = np.linspace(y.min(), y.max(), 100, endpoint=True)
#
#

    xp = points[hull.vertices, 0]
    yp = points[hull.vertices, 1]

    xp = np.append(xp, xp[0])
    yp = np.append(yp, yp[0])

    ax.plot(xp, yp, 'k-')
    #ax.plot(points[hull.vertices[0], 0], points[hull.vertices[[0, -1]], 1])

    #points = np.array([[x, y] for x, y in zip(x1, y1)])
    #hull = ConvexHull(points)
    #ax.plot(points[hull.vertices,0], points[hull.vertices,1])

    #zi = griddata((x, y), avg1, (xi, yi))

    #ax.contour(xi, yi, z, levels=1)


def gene_expr(data, tsne, gene, fig=None, ax=None, cmap=plt.cm.plasma, out=None):
    """
    Plot multiple genes on a grid.

    Parameters
    ----------
    data : Pandas dataframe
        Genes x samples expression matrix 
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    genes : array
        List of gene names
    """

    exp = get_gene_data(data, gene)

    return expr_plot(tsne, exp, fig=fig, ax=ax, cmap=cmap, out=out)
