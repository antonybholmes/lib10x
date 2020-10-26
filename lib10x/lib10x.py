# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:51:15 2018

@author: antony
"""
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import numpy as np
import scipy.sparse as sp_sparse
import tables
import pandas as pd
from sklearn.manifold import TSNE
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import kneighbors_graph
from scipy.interpolate import griddata
import h5py
from scipy.interpolate import interp1d

from scipy.spatial import distance
import networkx as nx
import os
import phenograph
import libplot
import libcluster
import libtsne
import seaborn as sns
from libsparse.libsparse import SparseDataFrame
from lib10x.sample import *
from scipy.spatial import ConvexHull
from PIL import Image, ImageFilter

from scipy.stats import binned_statistic

import imagelib

TNSE_AX_Q = 0.999

MARKER_SIZE = 10

SUBPLOT_SIZE = 4
EXP_ALPHA = 0.8
# '#f2f2f2' #(0.98, 0.98, 0.98) #(0.8, 0.8, 0.8) #(0.85, 0.85, 0.85
BACKGROUND_SAMPLE_COLOR = [0.75, 0.75, 0.75]
EDGE_COLOR = None  # [0.3, 0.3, 0.3] #'#4d4d4d'
EDGE_WIDTH = 0  # 0.25
ALPHA = 0.9

BLUE_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'blue_yellow', ['#162d50', '#ffdd55'])
BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'blue', ['#162d50', '#afc6e9'])
BLUE_GREEN_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bgy', ['#162d50', '#214478', '#217844', '#ffcc00', '#ffdd55'])

# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#002255', '#2ca05a', '#ffd42a'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#002255', '#003380', '#2ca05a', '#ffd42a', '#ffdd55'])

# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003366', '#339966', '#ffff66', '#ffff00')
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#001a33', '#003366', '#339933', '#ffff66', '#ffff00'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#00264d', '#003366', '#339933', '#e6e600', '#ffff33'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003366', '#40bf80', '#ffff33'])

BGY_ORIG_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bgy', ['#002255', '#003380', '#2ca05a', '#ffd42a', '#ffdd55'])

BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bgy', ['#003366', '#004d99', '#40bf80', '#ffe066', '#ffd633'])

GRAY_PURPLE_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'grey_purple_yellow', ['#e6e6e6', '#3333ff', '#ff33ff', '#ffe066'])

GYBLGRYL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'grey_blue_green_yellow', ['#e6e6e6', '#0055d4', '#00aa44', '#ffe066'])

OR_RED_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'or_red', matplotlib.cm.OrRd(range(4, 256)))

BU_PU_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bu_pu', matplotlib.cm.BuPu(range(4, 256)))


# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#0066ff', '#37c871', '#ffd42a'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003380', '#5fd38d', '#ffd42a'])

EXP_NORM = matplotlib.colors.Normalize(-1, 3, clip=True)

LEGEND_PARAMS = {'show': True, 'cols': 4, 'markerscale': 2}


CLUSTER_101_COLOR = (0.3, 0.3, 0.3)

np.random.seed(0)

GeneBCMatrix = collections.namedtuple(
    'GeneBCMatrix', ['gene_ids', 'gene_names', 'barcodes', 'matrix'])


def decode(items):
    return np.array([x.decode('utf-8') for x in items])


def get_matrix_from_h5(filename, genome):
    with tables.open_file(filename, 'r') as f:
        try:
            dsets = {}

            print(f.list_nodes('/'))

            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()

            # for node in f.walk_nodes('/matrix', 'Array'):
            #    dsets[node.name] = node.read()

            print(dsets)

            matrix = sp_sparse.csc_matrix(
                (dsets['data'], dsets['indices'], dsets['indptr']), shape=dsets['shape'])
            return GeneBCMatrix(decode(dsets['genes']), decode(dsets['gene_names']), decode(dsets['barcodes']), matrix)
        except tables.NoSuchNodeError:
            raise Exception("Genome %s does not exist in this file." % genome)
        except KeyError:
            raise Exception("File is missing one or more required datasets.")

#GeneBCMatrix = collections.namedtuple('FeatureBCMatrix', ['feature_ids', 'feature_names', 'barcodes', 'matrix'])


def get_matrix_from_h5_v2(filename, genome):
    with h5py.File(filename, 'r') as f:
        if u'version' in f.attrs:
            if f.attrs['version'] > 2:
                raise ValueError(
                    'Matrix HDF5 file format version (%d) is an newer version that is not supported by this function.' % version)
        else:
            raise ValueError(
                'Matrix HDF5 file format version (%d) is an older version that is not supported by this function.' % version)

        feature_ids = [x.decode('ascii', 'ignore')
                       for x in f['matrix']['features']['id']]
        feature_names = [x.decode('ascii', 'ignore')
                         for x in f['matrix']['features']['name']]
        barcodes = list(f['matrix']['barcodes'][:])
        matrix = sp_sparse.csc_matrix(
            (f['matrix']['data'], f['matrix']['indices'], f['matrix']['indptr']), shape=f['matrix']['shape'])
        return GeneBCMatrix(feature_ids, feature_names, decode(barcodes), matrix)


def save_matrix_to_h5(gbm, filename, genome):
    flt = tables.Filters(complevel=1)
    with tables.open_file(filename, 'w', filters=flt) as f:
        try:
            group = f.create_group(f.root, genome)
            f.create_carray(group, 'genes', obj=gbm.gene_ids)
            f.create_carray(group, 'gene_names', obj=gbm.gene_names)
            f.create_carray(group, 'barcodes', obj=gbm.barcodes)
            f.create_carray(group, 'data', obj=gbm.matrix.data)
            f.create_carray(group, 'indices', obj=gbm.matrix.indices)
            f.create_carray(group, 'indptr', obj=gbm.matrix.indptr)
            f.create_carray(group, 'shape', obj=gbm.matrix.shape)
        except:
            raise Exception("Failed to write H5 file.")


def subsample_matrix(gbm, barcode_indices):
    return GeneBCMatrix(gbm.gene_ids, gbm.gene_names, gbm.barcodes[barcode_indices], gbm.matrix[:, barcode_indices])


def get_expression(gbm, gene_name, genes=None):
    if genes is None:
        genes = gbm.gene_names

    gene_indices = np.where(genes == gene_name)[0]
    if len(gene_indices) == 0:
        raise Exception("%s was not found in list of gene names." % gene_name)
    return gbm.matrix[gene_indices[0], :].toarray().squeeze()


def gbm_to_df(gbm):
    return pd.DataFrame(gbm.matrix.todense(), index=gbm.gene_names, columns=gbm.barcodes)


def get_barcode_counts(gbm):
    ret = []
    for i in range(len(gbm.barcodes)):
        ret.append(np.sum(gbm.matrix[:, i].toarray()))

    return ret


def df(gbm):
    """ 
    Converts a GeneBCMatrix to a pandas dataframe (dense)

    Parameters
    ----------
    gbm : a GeneBCMatrix

    Returns
    -------
    object : Pandas DataFrame shape(n_cells, n_genes)
    """

    df = pd.DataFrame(gbm.matrix.todense())
    df.index = gbm.gene_names
    df.columns = gbm.barcodes

    return df


def to_csv(gbm, file, sep='\t'):
    df(gbm).to_csv(file, sep=sep, header=True, index=True)


def sum(gbm, axis=0):
    return gbm.matrix.sum(axis=axis)


def tpm(gbm):
    m = gbm.matrix
    s = 1 / m.sum(axis=0)
    mn = m.multiply(s)
    tpm = mn.multiply(1000000)

    return GeneBCMatrix(gbm.gene_ids, gbm.gene_names, gbm.barcodes, tpm)


def create_cluster_plots(pca, labels, name, marker='o', s=MARKER_SIZE):
    for i in range(0, pca.shape[1]):
        for j in range(i + 1, pca.shape[1]):
            create_cluster_plot(pca, labels, name, pc1=(
                i + 1), pc2=(j + 1), marker=marker, s=s)


def pca_base_plots(pca, clusters, n=10, marker='o', s=MARKER_SIZE):
    rows = libplot.grid_size(n)

    w = 4 * rows

    fig = libplot.new_base_fig(w=w, h=w)

    si = 1

    for i in range(0, n):
        for j in range(i + 1, n):
            ax = libplot.new_ax(fig, subplot=(rows, rows, si))

            pca_plot_base(pca, clusters, pc1=(i + 1),
                          pc2=(j + 1), marker=marker, s=s, ax=ax)

            si += 1

    return fig


def pca_plot_base(pca,
                  clusters,
                  pc1=1,
                  pc2=2,
                  marker='o',
                  labels=False,
                  s=MARKER_SIZE,
                  w=8,
                  h=8,
                  fig=None,
                  ax=None):
    colors = libcluster.get_colors()

    if ax is None:
        fig, ax = libplot.new_fig(w=w, h=h)

    ids = list(sorted(set(clusters['Cluster'])))

    for i in range(0, len(ids)):
        l = ids[i]

        #print('Label {}'.format(l))
        indices = np.where(clusters['Cluster'] == l)[0]

        n = len(indices)

        label = 'C{} ({:,})'.format(l, n)

        df2 = pca.iloc[indices, ]

        x = df2.iloc[:, pc1 - 1]
        y = df2.iloc[:, pc2 - 1]

        if i in colors:
            color = colors[i]  # l]
        else:
            color = 'black'

        ax.scatter(x, y, color=color, edgecolor=color, s=s,
                   marker=marker, alpha=libplot.ALPHA, label=label)

    if labels:
        l = pca.index.values
        for i in range(0, pca.shape[0]):
            print(pca.shape, pca.iloc[i, pc1 - 1], pca.iloc[i, pc2 - 1])
            ax.text(pca.iloc[i, pc1 - 1], pca.iloc[i, pc2 - 1], pca.index[i])

    return fig, ax


def pca_plot(pca,
             clusters,
             pc1=1,
             pc2=2,
             marker='o',
             labels=False,
             s=MARKER_SIZE,
             w=8,
             h=8,
             legend=True,
             fig=None,
             ax=None):
    fig, ax = pca_plot_base(pca,
                            clusters,
                            pc1=pc1,
                            pc2=pc2,
                            marker=marker,
                            labels=labels,
                            s=s,
                            w=w,
                            h=h,
                            fig=fig,
                            ax=ax)

    #libtsne.tsne_legend(ax, labels, colors)
    libcluster.format_simple_axes(ax, title="PC")

    if legend:
        libcluster.format_legend(ax, cols=6, markerscale=2)

    return fig, ax


def create_pca_plot(pca,
                    clusters,
                    name,
                    pc1=1,
                    pc2=2,
                    marker='o',
                    labels=False,
                    legend=True,
                    s=MARKER_SIZE,
                    w=8,
                    h=8,
                    fig=None,
                    ax=None,
                    dir='.',
                    format='png'):
    out = '{}/pca_{}_pc{}_vs_pc{}.{}'.format(dir, name, pc1, pc2, format)

    fig, ax = pca_plot(pca,
                       clusters,
                       pc1=pc1,
                       pc2=pc2,
                       labels=labels,
                       marker=marker,
                       legend=legend,
                       s=s,
                       w=w,
                       h=h,
                       fig=fig,
                       ax=ax)

    libplot.savefig(fig, out, pad=2)
    plt.close(fig)


def set_tsne_ax_lim(tsne, ax):
    """
    Set the t-SNE x,y limits to look pretty.
    """

    d1 = tsne.iloc[:, 0]
    d2 = tsne.iloc[:, 1]

    xlim = [d1[d1 < 0].quantile(1 - TNSE_AX_Q),
            d1[d1 >= 0].quantile(TNSE_AX_Q)]
    ylim = [d2[d2 < 0].quantile(1 - TNSE_AX_Q),
            d2[d2 >= 0].quantile(TNSE_AX_Q)]

    #print(xlim, ylim)

    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)


def base_cluster_plot(d,
                      clusters,
                      markers=None,
                      s=libplot.MARKER_SIZE,
                      colors=None,
                      edgecolors=EDGE_COLOR,
                      linewidth=EDGE_WIDTH,
                      dim1=0,
                      dim2=1,
                      w=8,
                      h=8,
                      alpha=ALPHA,  # libplot.ALPHA,
                      show_axes=True,
                      legend=True,
                      sort=True,
                      cluster_order=None,
                      fig=None,
                      ax=None):
    """
    Create a tsne plot without the formatting

    Parameters
    ----------
    d : Pandas dataframe
        t-sne, umap data
    clusters : Pandas dataframe
        n x 1 table of n cells with a Cluster column giving each cell a
        cluster label.
    s : int, optional
        Marker size
    w : int, optional
        Plot width
    h : int, optional
        Plot height
    alpha : float (0, 1), optional
        Tranparency of markers.
    show_axes : bool, optional, default true
        Whether to show axes on plot
    legend : bool, optional, default true
        Whether to show legend.
    """

    if ax is None:
        fig, ax = libplot.new_fig(w=w, h=h)

    libcluster.scatter_clusters(d.iloc[:, dim1].values,
                                d.iloc[:, dim2].values,
                                clusters,
                                colors=colors,
                                edgecolors=edgecolors,
                                linewidth=linewidth,
                                markers=markers,
                                alpha=alpha,
                                s=s,
                                ax=ax,
                                cluster_order=cluster_order,
                                sort=sort)

    #set_tsne_ax_lim(tsne, ax)

    # libcluster.format_axes(ax)

    if not show_axes:
        libplot.invisible_axes(ax)

    legend_params = dict(LEGEND_PARAMS)

    if isinstance(legend, bool):
        legend_params['show'] = legend
    elif isinstance(legend, dict):
        legend_params.update(legend)
    else:
        pass

    if legend_params['show']:
        libcluster.format_legend(ax,
                                 cols=legend_params['cols'],
                                 markerscale=legend_params['markerscale'])

    return fig, ax


def base_cluster_plot_outline(out,
                              d,
                              clusters,
                              s=libplot.MARKER_SIZE,
                              colors=None,
                              edgecolors=EDGE_COLOR,
                              linewidth=EDGE_WIDTH,
                              dim1=0,
                              dim2=1,
                              w=8,
                              alpha=ALPHA,  # libplot.ALPHA,
                              show_axes=True,
                              legend=True,
                              sort=True,
                              outline=True):

    cluster_order = list(sorted(set(clusters['Cluster'])))

    im_base = imagelib.new(w * 300, w * 300)

    for i in range(0, len(cluster_order)):
        print('index', i, cluster_order[i])
        cluster = cluster_order[i]

        if isinstance(colors, dict):
            color = colors[cluster]
        elif isinstance(colors, list):
            if cluster < len(colors):
                # np.where(clusters['Cluster'] == cluster)[0]]
                color = colors[i]
            else:
                color = 'black'
        else:
            color = 'black'

        fig, ax = separate_cluster(d,
                                   clusters,
                                   cluster,
                                   color=color,
                                   size=w,
                                   s=s,
                                   linewidth=linewidth,
                                   add_titles=False)
        # get x y lim
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        fig, ax = separate_cluster(d,
                                   clusters,
                                   cluster,
                                   color=color,
                                   size=w,
                                   s=s,
                                   linewidth=linewidth,
                                   add_titles=False,
                                   show_background=False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if not show_axes:
            libplot.invisible_axes(ax)

        legend_params = dict(LEGEND_PARAMS)

        if isinstance(legend, bool):
            legend_params['show'] = legend
        elif isinstance(legend, dict):
            legend_params.update(legend)
        else:
            pass

        if legend_params['show']:
            libcluster.format_legend(ax,
                                     cols=legend_params['cols'],
                                     markerscale=legend_params['markerscale'])

        libplot.invisible_axes(ax)

        tmp = 'tmp{}.png'.format(i)

        libplot.savefig(fig, tmp)

        plt.close(fig)

        # Open image
#        im = imagelib.open(tmp)
#        im_no_bg = imagelib.remove_background(im)
#        im_smooth = imagelib.smooth_edges(im_no_bg)
#        imagelib.paste(im_no_bg, im_smooth, inplace=True)
#        imagelib.save(im_no_bg, 'smooth.png')
#        imagelib.paste(im_base, im_no_bg, inplace=True)

        im = imagelib.open(tmp)

        if outline:
            im_no_bg = imagelib.remove_background(im)
            im_edges = imagelib.edges(im)
            im_outline = imagelib.paste(im, im_edges)  # im_no_bg
            im_smooth = imagelib.smooth(im_outline)

            imagelib.save(im_smooth, 'smooth.png')  # im_smooth
            imagelib.paste(im_base, im_smooth, inplace=True)
        else:
            imagelib.paste(im_base, im, inplace=True)

#        # find gray areas and mask
#        im_data = np.array(im1.convert('RGBA'))
#
#        r = im_data[:, :, 0]
#        g = im_data[:, :, 1]
#        b = im_data[:, :, 2]
#
#        grey_areas = (r < 255) & (r > 200) & (g < 255) & (g > 200) & (b < 255) & (b > 200)
#
#        d = im_data[np.where(grey_areas)]
#        d[:, :] = [255, 255, 255, 0]
#        im_data[np.where(grey_areas)] = d
#
#        im2 = Image.fromarray(im_data)
#
#        # Edge detect on what is left (the clusters)
#        im_edges = im2.filter(ImageFilter.FIND_EDGES)
#
#        im_smooth = im_edges.filter(ImageFilter.SMOOTH)
#
#        # paste outline onto clusters
#        im2.paste(im_smooth, (0, 0), im_smooth)
#
#        # overlay edges on top of original image to highlight cluster
#        im_base.paste(im2, (0, 0), im2)
        # break

    imagelib.save(im_base, out)


def cluster_plot(tsne,
                 clusters,
                 dim1=0,
                 dim2=1,
                 markers='o',
                 s=libplot.MARKER_SIZE,
                 colors=None,
                 w=8,
                 h=8,
                 legend=True,
                 show_axes=False,
                 sort=True,
                 cluster_order=None,
                 fig=None,
                 ax=None,
                 out=None):
    fig, ax = base_cluster_plot(tsne,
                                clusters,
                                markers=markers,
                                colors=colors,
                                dim1=dim1,
                                dim2=dim2,
                                s=s,
                                w=w,
                                h=h,
                                cluster_order=cluster_order,
                                legend=legend,
                                sort=sort,
                                show_axes=show_axes,
                                fig=fig,
                                ax=ax)

    #libtsne.tsne_legend(ax, labels, colors)
    #libcluster.format_simple_axes(ax, title="t-SNE")
    #libcluster.format_legend(ax, cols=6, markerscale=2)

    if out is not None:
        libplot.savefig(fig, out)

    return fig, ax


def create_cluster_plot(d,
                        clusters,
                        name,
                        dim1=0,
                        dim2=1,
                        method='tsne',
                        markers='o',
                        s=libplot.MARKER_SIZE,
                        w=8,
                        h=8,
                        colors=None,
                        legend=True,
                        sort=True,
                        show_axes=False,
                        ax=None,
                        cluster_order=None,
                        format='png',
                        dir='.',
                        out=None):

    if out is None:
        # libtsne.get_tsne_plot_name(name))
        out = '{}/{}_{}.{}'.format(dir, method, name, format)

    print(out)

    return cluster_plot(d,
                        clusters,
                        dim1=dim1,
                        dim2=dim2,
                        markers=markers,
                        colors=colors,
                        s=s,
                        w=w,
                        h=h,
                        cluster_order=cluster_order,
                        show_axes=show_axes,
                        legend=legend,
                        sort=sort,
                        out=out)


def base_tsne_plot(tsne, marker='o', s=libplot.MARKER_SIZE, c='red', label=None, fig=None, ax=None):
    """
    Create a tsne plot without the formatting
    """

    if ax is None:
        fig, ax = libplot.new_fig()

    libplot.scatter(tsne['TSNE-1'], tsne['TSNE-2'], c=c,
                    marker=marker, label=label, s=s, ax=ax)

    return fig, ax


def tsne_plot(tsne, marker='o', s=libplot.MARKER_SIZE, c='red', label=None, fig=None, ax=None):
    fig, ax = base_tsne_plot(tsne, marker=marker, c=c,
                             s=s, label=label, fig=fig, ax=ax)

    #libtsne.tsne_legend(ax, labels, colors)
    libcluster.format_simple_axes(ax, title="t-SNE")
    libcluster.format_legend(ax, cols=6, markerscale=2)

    return fig, ax


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


def get_gene_names(data):
    if ';' in data.index[0]:
        ids, genes = data.index.str.split(';').str
    else:
        genes = data.index
        ids = genes

    return ids.values, genes.values


def get_gene_ids(data, genes, ids=None, gene_names=None):
    """
    For a given gene list, get all of the transcripts.

    Parameters
    ----------
    data : DataFrame
        data table containing and index
    genes : list
        List of strings of gene ids
    ids : Index, optional
        Index of gene ids
    gene_names : Index, optional
        Index of gene names

    Returns
    -------
    list
        list of tuples of (index, gene_id, gene_name)
    """

    if ids is None:
        ids, gene_names = get_gene_names(data)

    ret = []

    for g in genes:
        indexes = np.where(ids == g)[0]

        if indexes.size > 0:
            for index in indexes:
                ret.append((index, ids[index], gene_names[index]))
        else:
            # if id does not exist, try the gene names
            indexes = np.where(gene_names == g)[0]

            for index in indexes:
                ret.append((index, ids[index], gene_names[index]))

    return ret


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
        #    libcluster.format_axes(ax)
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


        libplot.savefig(fig, 'tmp.png', pad=0)
        libplot.savefig(fig, out, pad=0)
        plt.close(fig)

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


def separate_cluster(tsne,
                     clusters,
                     cluster,
                     color='black',
                     background=BACKGROUND_SAMPLE_COLOR,
                     bgedgecolor='#808080',
                     show_background=True,
                     add_titles=True,
                     size=4,
                     alpha=ALPHA,
                     s=MARKER_SIZE,
                     edgecolors='white',
                     linewidth=EDGE_WIDTH,
                     fig=None,
                     ax=None):
    """
    Plot a cluster separately to highlight where the samples are

    Parameters
    ----------
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    cluster : int
        Clusters in 
    colors : list, color
        Colors of points
    add_titles : bool
        Whether to add titles to plots
    w: int, optional
        width of new ax.
    h: int, optional
        height of new ax.

    Returns
    -------
    fig : Matplotlib figure
        A new Matplotlib figure used to make the plot
    ax : Matplotlib axes
        Axes used to render the figure
    """

    if ax is None:
        fig, ax = libplot.new_fig(size, size)

    #print('Label {}'.format(l))
    idx1 = np.where(clusters['Cluster'] == cluster)[0]
    idx2 = np.where(clusters['Cluster'] != cluster)[0]

    # Plot background points
    if show_background:
        x = tsne.iloc[idx2, 0]
        y = tsne.iloc[idx2, 1]

        libplot.scatter(x,
                        y,
                        c=[background],
                        ax=ax,
                        edgecolors='none',  # bgedgecolor,
                        linewidth=linewidth,
                        s=s)

    # Plot cluster over the top of the background

    x = tsne.iloc[idx1, 0]
    y = tsne.iloc[idx1, 1]

    #print('sep', cluster, color)

    color = color  # + '7f'

    libplot.scatter(x,
                    y,
                    c=color,
                    ax=ax,
                    edgecolors='none',  # edgecolors,
                    linewidth=linewidth,
                    s=s)

    if add_titles:
        if isinstance(cluster, int):
            prefix = 'C'
        else:
            prefix = ''

        ax.set_title('{}{} ({:,})'.format(
            prefix, cluster, len(idx1)), color=color)

    ax.axis('off')  # libplot.invisible_axes(ax)

    return fig, ax


def separate_clusters(tsne,
                      clusters,
                      name,
                      colors=None,
                      size=4,
                      add_titles=True,
                      type='tsne',
                      format='pdf'):
    """
    Plot each cluster into its own plot file.
    """

    ids = list(sorted(set(clusters['Cluster'])))

    indices = np.array(list(range(0, len(ids))))

    if colors is None:
        colors = libcluster.get_colors()

    for i in indices:
        print('index', i)
        cluster = ids[i]

        if isinstance(colors, dict):
            color = colors[cluster]
        elif isinstance(colors, list):
            if cluster < len(colors):
                # np.where(clusters['Cluster'] == cluster)[0]]
                color = colors[i]
            else:
                color = CLUSTER_101_COLOR
        else:
            color = 'black'

        fig, ax = separate_cluster(tsne,
                                   clusters,
                                   cluster,
                                   color=color,
                                   add_titles=add_titles,
                                   size=size)

        out = '{}_sep_clust_{}_c{}.{}'.format(type, name, cluster, format)

        print('Creating', out, '...')

        libplot.savefig(fig, out)
        libplot.savefig(fig, 'tmp.png')
        plt.close(fig)


def cluster_grid(tsne,
                 clusters,
                 colors=None,
                 cols=-1,
                 size=SUBPLOT_SIZE,
                 add_titles=True,
                 cluster_order=None):
    """
    Plot each cluster separately to highlight where the samples are

    Parameters
    ----------
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    clusters : DataFrame
        Clusters in 
    colors : list, color
        Colors of points
    add_titles : bool
        Whether to add titles to plots
    plot_order: list, optional
        List of cluster ids in the order they should be rendered

    Returns
    -------
    fig : Matplotlib figure
        A new Matplotlib figure used to make the plot
    """

    if cluster_order is None:
        ids = np.array(list(sorted(set(clusters['Cluster']))))
        cluster_order = np.array(list(range(0, len(ids)))) + 1

    n = cluster_order.size

    if cols == -1:
        cols = int(np.ceil(np.sqrt(n)))

    rows = int(np.ceil(n / cols))

    w = size * cols
    h = size * rows

    fig = libplot.new_base_fig(w=w, h=h)

    if colors is None:
        colors = libcluster.get_colors()

    # Where to plot figure
    pc = 1

    for c in cluster_order:
        i = c - 1
        cluster = ids[i]
        # look up index for color purposes
        #i = np.where(ids == cluster)[0][0]

        print('index', i, cluster, colors)

        if isinstance(colors, dict):
            color = colors.get(cluster, 'black')
        elif isinstance(colors, list):
            #i = cluster - 1
            if i < len(colors):

                # colors[cid - 1] #colors[i] #np.where(clusters['Cluster'] == cluster)[0]]
                color = colors[i]
            else:
                color = 'black'
        else:
            color = 'black'

        ax = libplot.new_ax(fig, subplot=(rows, cols, pc))

        separate_cluster(tsne,
                         clusters,
                         cluster,
                         color=color,
                         add_titles=add_titles,
                         ax=ax)

#        idx1 = np.where(clusters['Cluster'] == cluster)[0]
#        idx2 = np.where(clusters['Cluster'] != cluster)[0]
#
#        # Plot background points
#
#
#
#        x = tsne.iloc[idx2, 0]
#        y = tsne.iloc[idx2, 1]
#        libplot.scatter(x, y, c=BACKGROUND_SAMPLE_COLOR, ax=ax)
#
#        # Plot cluster over the top of the background
#
#        x = tsne.iloc[idx1, 0]
#        y = tsne.iloc[idx1, 1]
#
#        if isinstance(colors, dict):
#            color = colors[cluster]
#        elif isinstance(colors, list):
#            color = colors[i]
#        else:
#            color = 'black'
#
#        libplot.scatter(x, y, c=color, ax=ax)
#
#        if add_titles:
#            if isinstance(cluster, int):
#                prefix = 'C'
#            else:
#                prefix = ''
#
#            ax.set_title('{}{} ({:,})'.format(prefix, cluster, len(idx1)), color=color)
#
#
#        libplot.invisible_axes(ax)

        pc += 1

    return fig


def create_cluster_grid(tsne,
                        clusters,
                        name,
                        colors=None,
                        cols=-1,
                        size=SUBPLOT_SIZE,
                        add_titles=True,
                        cluster_order=None,
                        method='tsne',
                        dir='.',
                        out=None):
    fig = cluster_grid(tsne,
                       clusters,
                       colors=colors,
                       cols=cols,
                       size=size,
                       add_titles=add_titles,
                       cluster_order=cluster_order)

    if out is None:
        out = '{}/{}_{}_separate_clusters.png'.format(dir, method, name)

    libplot.savefig(fig, out, pad=0)
    #libplot.savefig(fig, '{}/tsne_{}separate_clusters.pdf'.format(dir, name))


#
#
# def tsne_cluster_sample_grid(tsne, clusters, samples, colors=None, size=SUBPLOT_SIZE):
#    """
#    Plot each cluster separately to highlight samples
#
#    Parameters
#    ----------
#    tsne : Pandas dataframe
#        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
#    clusters : DataFrame
#        Clusters in
#
#    Returns
#    -------
#    fig : Matplotlib figure
#        A new Matplotlib figure used to make the plot
#    """
#
#
#    cids = list(sorted(set(clusters['Cluster'])))
#
#    rows = int(np.ceil(np.sqrt(len(cids))))
#
#    w = size * rows
#
#    fig = libplot.new_base_fig(w=w, h=w)
#
#    if colors is None:
#        colors = libcluster.colors()
#
#    for i in range(0, len(cids)):
#        c = cids[i]
#
#        #print('Label {}'.format(l))
#        idx2 = np.where(clusters['Cluster'] != c)[0]
#
#        # Plot background points
#
#        ax = libplot.new_ax(fig, subplot=(rows, rows, i + 1))
#
#        x = tsne.iloc[idx2, 0]
#        y = tsne.iloc[idx2, 1]
#
#        libplot.scatter(x, y, c=BACKGROUND_SAMPLE_COLOR, ax=ax)
#
#        # Plot cluster over the top of the background
#
#        sid = 0
#
#        for sample in samples:
#            id = '-{}'.format(sid + 1)
#            idx1 = np.where((clusters['Cluster'] == c) & clusters.index.str.contains(id))[0]
#
#            x = tsne.iloc[idx1, 0]
#            y = tsne.iloc[idx1, 1]
#
#            libplot.scatter(x, y, c=colors[sid], ax=ax)
#
#            sid += 1
#
#        ax.set_title('C{} ({:,})'.format(c, len(idx1)), color=colors[i])
#        libplot.invisible_axes(ax)
#
#        #set_tsne_ax_lim(tsne, ax)
#
#    return fig
#
#
# def create_tsne_cluster_sample_grid(tsne, clusters, samples, name, colors=None, size=SUBPLOT_SIZE, dir='.'):
#    """
#    Plot separate clusters colored by sample
#    """
#    fig = tsne_cluster_sample_grid(tsne, clusters, samples, colors, size)
#
#    libplot.savefig(fig, '{}/tsne_{}_sample_clusters.png'.format(dir, name))
#    #libplot.savefig(fig, '{}/tsne_{}separate_clusters.pdf'.format(dir, name))
#
#


def load_clusters(pca, headers, name, cache=True):
    file = libtsne.get_cluster_file(name)

    if not os.path.isfile(file) or not cache:
        print('{} was not found, creating it with...'.format(file))

        # Find the interesting clusters
        labels, graph, Q = phenograph.cluster(pca, k=20)

        if min(labels) == -1:
            new_label = 100
            labels[np.where(labels == -1)] = new_label

        labels += 1

        libtsne.write_clusters(headers, labels, name)

    cluster_map, data = libtsne.read_clusters(file)

    labels = data  # .tolist()

    return cluster_map, labels


def umi_tpm(data):
    # each column is a cell
    reads_per_bc = data.sum(axis=0)
    scaling_factors = 1000000 / reads_per_bc
    scaled = data.multiply(scaling_factors)  # , axis=1)
    return scaled


def umi_log2(d):
    if isinstance(d, SparseDataFrame):
        print('UMI norm log2 sparse')

        return d.log2(add=1)
    else:
        return (d + 1).apply(np.log2)


def umi_tpm_log2(data):
    d = umi_tpm(data)

    return umi_log2(d)


def umi_norm(data):
    """
    Scale each library to its median size

    Parameters
    ----------
    data : Pandas dataframe
        Matrix of umi counts

    """

    # each column is a cell
    reads_per_bc = data.sum(axis=0)
    # int(np.round(np.median(reads_per_bc)))
    median_reads_per_bc = np.median(reads_per_bc)
    scaling_factors = median_reads_per_bc / reads_per_bc
    scaled = data.multiply(scaling_factors)  # , axis=1)
    return scaled


def umi_norm_log2(data):
    d = umi_norm(data)

    print(type(d))

    return umi_log2(d)


def scale(d, clip=None, min=None, max=None, axis=1):
    if isinstance(d, SparseDataFrame):
        print('UMI norm log2 scale sparse')
        sd = StandardScaler(with_mean=False).fit_transform(d.T.matrix)

        return SparseDataFrame(sd.T, index=d.index, columns=d.columns)
    else:
        # StandardScaler().fit_transform(d.T)
        sd = sklearn.preprocessing.scale(d, axis=axis)

        #sd = sd.T

        if isinstance(clip, float) or isinstance(clip, int):
            max = abs(clip)
            min = -max

        if isinstance(min, float) or isinstance(min, int):
            print('z min', min)
            sd[np.where(sd < min)] = min

        if isinstance(max, float) or isinstance(max, int):
            print('z max', max)
            sd[np.where(sd > max)] = max

        return pd.DataFrame(sd, index=d.index, columns=d.columns)


def min_max_scale(d, min=0, max=1, axis=1):
    #m = d.min(axis=1)
    #std = (d - m) / (d.max(axis=1) - m)
    #scaled = std * (max - min) + min

    # return scaled

    if axis == 0:
        return pd.DataFrame(MinMaxScaler(feature_range=(min, max)).fit_transform(d), index=d.index, columns=d.columns)
    else:
        return pd.DataFrame(MinMaxScaler(feature_range=(min, max)).fit_transform(d.T).T, index=d.index, columns=d.columns)


def rscale(d, min=0, max=1, axis=1):
    if axis == 0:
        return pd.DataFrame(RobustScaler().fit_transform(d), index=d.index, columns=d.columns)
    else:
        return pd.DataFrame(RobustScaler().fit_transform(d.T).T, index=d.index, columns=d.columns)


def umi_norm_log2_scale(data, clip=None):
    d = umi_norm_log2(data)

    return scale(d, clip=clip)


def read_clusters(file):
    print('Reading clusters from {}...'.format(file))

    return pd.read_csv(file, sep='\t', header=0, index_col=0)


def silhouette(tsne, tsne_umi_log2, clusters, name):
    # measure cluster worth
    x1 = silhouette_samples(
        tsne, clusters.iloc[:, 0].tolist(), metric='euclidean')
    x2 = silhouette_samples(
        tsne_umi_log2, clusters.iloc[:, 0].tolist(), metric='euclidean')

    fig, ax = libplot.newfig(w=9, h=7, subplot=211)
    df = pd.DataFrame({'Silhouette Score': x1, 'Cluster': clusters.iloc[:, 0].tolist(
    ), 'Label': np.repeat('tsne-10x', len(x1))})
    libplot.boxplot(df, 'Cluster', 'Silhouette Score',
                    colors=libcluster.colors(), ax=ax)
    ax.set_ylim([-1, 1])
    ax.set_title('tsne-10x')
    #libplot.savefig(fig, 'RK10001_10003_clust-phen_silhouette.pdf')

    ax = fig.add_subplot(212)  # libplot.newfig(w=9)
    df2 = pd.DataFrame({'Silhouette Score': x2, 'Cluster': clusters.iloc[:, 0].tolist(
    ), 'Label': np.repeat('tsne-ah', len(x2))})
    libplot.boxplot(df2, 'Cluster', 'Silhouette Score',
                    colors=libcluster.colors(), ax=ax)
    ax.set_ylim([-1, 1])
    ax.set_title('tsne-ah')
    libplot.savefig(fig, '{}_silhouette.pdf'.format(name))


def node_color_from_cluster(clusters):
    colors = libcluster.colors()

    return [colors[clusters['Cluster'][i] - 1] for i in range(0, clusters.shape[0])]

# def network(tsne, clusters, name, k=5):
#    A = kneighbors_graph(tsne, k, mode='distance', metric='euclidean').toarray()
#
#    #A = A[0:500, 0:500]
#
#    G=nx.from_numpy_matrix(A)
#    pos=nx.spring_layout(G) #, k=2)
#
#    #node_color = (c_phen['Cluster'][0:A.shape[0]] - 1).tolist()
#    node_color = (clusters['Cluster'] - 1).tolist()
#
#    fig, ax = libplot.newfig(w=10, h=10)
#
#    nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, node_size=50, node_color=node_color, vmax=(clusters['Cluster'].max() - 1), cmap=libcluster.colormap())
#
#    libplot.savefig(fig, 'network_{}.pdf'.format(name))


def plot_centroids(tsne, clusters, name):
    c = centroids(tsne, clusters)

    fig, ax = libplot.newfig(w=5, h=5)
    ax.scatter(c[:, 0], c[:, 1], c=None)
    libplot.format_axes(ax)
    libplot.savefig(fig, '{}_centroids.pdf'.format(name))


def centroid_network(tsne, clusters, name):
    c = centroids(tsne, clusters)

    A = kneighbors_graph(c, 5, mode='distance', metric='euclidean').toarray()
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)

    fig, ax = libplot.newfig(w=8, h=8)
    # list(range(0, c.shape[0]))
    node_color = libcluster.colors()[0:c.shape[0]]
    cmap = libcluster.colormap()

    labels = {}

    for i in range(0, c.shape[0]):
        labels[i] = i + 1

    #nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, node_size=200, node_color=node_color, vmax=(c.shape[0] - 1), cmap=libcluster.colormap())
    nx.draw_networkx(G, with_labels=True, labels=labels, ax=ax, node_size=800,
                     node_color=node_color, font_color='white', font_family='Arial')

    libplot.format_axes(ax)
    libplot.savefig(fig, '{}_centroid_network.pdf'.format(name))


def centroids(tsne, clusters):
    cids = list(sorted(set(clusters['Cluster'].tolist())))

    ret = np.zeros((len(cids), 2))

    for i in range(0, len(cids)):
        c = cids[i]
        x = tsne.iloc[np.where(clusters['Cluster'] == c)[0], :]
        centroid = (x.sum(axis=0) / x.shape[0]).tolist()
        ret[i, 0] = centroid[0]
        ret[i, 1] = centroid[1]

    return ret


def knn_method_overlaps(tsne1, tsne2, clusters, name, k=5):
    c1 = centroids(tsne1, clusters)
    c2 = centroids(tsne2, clusters)

    a1 = kneighbors_graph(c1, k, mode='distance', metric='euclidean').toarray()
    a2 = kneighbors_graph(c2, k, mode='distance', metric='euclidean').toarray()

    overlaps = []

    for i in range(0, c1.shape[0]):
        ids1 = np.where(a1[i, :] > 0)[0]
        ids2 = np.where(a2[i, :] > 0)[0]
        ids3 = np.intersect1d(ids1, ids2)
        o = len(ids3) / 5 * 100
        overlaps.append(o)

    df = pd.DataFrame(
        {'Cluster': list(range(1, c1.shape[0] + 1)), 'Overlap %': overlaps})
    df.set_index('Cluster', inplace=True)
    df.to_csv('{}_cluster_overlaps.txt'.format(name), sep='\t')


def mkdir(path):
    """
    Make dirs including any parents and avoid raising exception to work
    more like mkdir -p

    Parameters
    ----------
    path : str
        directory to create.

    """

    try:
        os.makedirs(path)
    except:
        pass


def split_a_b(counts, samples, w=6, h=6, format='pdf'):
    """ 
    Split cells into a and b
    """
    cache = True

    counts = libcluster.remove_empty_rows(counts)

    # ['AICDA', 'CD83', 'CXCR4', 'MKI67', 'MYC', 'PCNA', 'PRDM1']
    genes = pd.read_csv('../../../../expression_genes.txt', header=0)

    mkdir('a')

    a_barcodes = pd.read_csv('../a_barcodes.tsv', header=0, sep='\t')
    idx = np.where(counts.columns.isin(a_barcodes['Barcode'].values))[0]
    d_a = counts.iloc[:, idx]
    d_a = libcluster.remove_empty_rows(d_a)

    if isinstance(d_a, SparseDataFrame):
        d_a = umi_norm_log2(d_a)
    else:
        d_a = umi_norm_log2_scale(d_a)

    pca_a = libtsne.load_pca(d_a, 'a', cache=cache)  # pca.iloc[idx,:]
    tsne_a = libtsne.load_pca_tsne(pca_a, 'a', cache=cache)
    c_a = libtsne.load_phenograph_clusters(pca_a, 'a', cache=cache)

    create_pca_plot(pca_a, c_a, 'a', dir='a')
    create_cluster_plot(tsne_a, c_a, 'a', dir='a')
    create_cluster_grid(tsne_a, c_a, 'a', dir='a')

    create_merge_cluster_info(d_a, c_a, 'a', sample_names=samples, dir='a')
    create_cluster_samples(tsne_a, c_a, samples, 'a_sample', dir='a')

    genes_expr(d_a, tsne_a, genes, prefix='a_BGY',
               cmap=BLUE_GREEN_YELLOW_CMAP, w=w, h=h, dir='a/GeneExp', format=format)

    fig, ax = cluster_plot(tsne_a, c_a, legend=False, w=w, h=h)
    libplot.savefig(fig, 'a/a_tsne_clusters_med.pdf')

    # b

    mkdir('b')

    b_barcodes = pd.read_csv('../b_barcodes.tsv', header=0, sep='\t')
    idx = np.where(counts.columns.isin(b_barcodes['Barcode'].values))[0]
    d_b = counts.iloc[:, idx]
    d_b = libcluster.remove_empty_rows(d_b)

    if isinstance(d_a, SparseDataFrame):
        d_b = umi_norm_log2(d_b)
    else:
        d_b = umi_norm_log2_scale(d_b)

    pca_b = libtsne.load_pca(d_b, 'b', cache=cache)  # pca.iloc[idx_b,:]
    tsne_b = libtsne.load_pca_tsne(pca_b, 'b', cache=cache)
    c_b = libtsne.load_phenograph_clusters(pca_b, 'b', cache=cache)

    create_pca_plot(pca_b, c_b, 'b', dir='b')
    create_cluster_plot(tsne_b, c_b, 'b', dir='b')
    create_cluster_grid(tsne_b, c_b, 'b', dir='b')
    create_merge_cluster_info(d_b, c_b, 'b', sample_names=samples, dir='b')
    create_cluster_samples(tsne_b, c_b, samples, 'b_sample', dir='b')

    genes_expr(d_b, tsne_b, genes, prefix='b_BGY',
               cmap=BLUE_GREEN_YELLOW_CMAP, w=w, h=h, dir='b/GeneExp', format=format)

    fig, ax = cluster_plot(tsne_b, c_b, legend=False, w=w, h=h)
    libplot.savefig(fig, 'b/b_tsne_clusters_med.pdf')


def sample_clusters(d, sample_names):
    """
    Create a cluster matrix based on by labelling cells by sample/batch.
    """

    sc = np.array(['' for i in range(0, d.shape[0])], dtype=object)

    c = 1

    for s in sample_names:
        id = '-{}'.format(c)
        print(id)
        print(np.where(d.index.str.contains(id))[0])
        sc[np.where(d.index.str.contains(id))[0]] = s
        c += 1

    print(np.unique(d.index.values))
    print(np.unique(sc))

    df = pd.DataFrame(sc, index=d.index, columns=['Cluster'])

    return df


def create_cluster_samples(tsne_umi_log2,
                           clusters,
                           sample_names,
                           name,
                           method='tsne',
                           format='png',
                           dir='.',
                           w=16,
                           h=16,
                           legend=True):
    sc = sample_clusters(clusters, sample_names)

    create_cluster_plot(tsne_umi_log2,
                        sc,
                        name,
                        method=method,
                        format=format,
                        dir=dir,
                        w=w,
                        h=w,
                        legend=legend)
