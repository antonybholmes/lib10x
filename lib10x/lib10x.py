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

from .outline import cluster_outline

import imagelib

from .constants import *


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


def separate_cluster(tsne,
                     clusters,
                     cluster,
                     color='black',
                     background=BACKGROUND_SAMPLE_COLOR,
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

    cluster_outline(tsne,
             cluster,
             clusters,
             sdmax=0.5,
             ax=ax)

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
    cids = list(sorted(set(clusters['Cluster'].values)))

    ret = np.zeros((len(cids), 2))

    for i, c in enumerate(cids):
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


def sample_clusters(clusters, sample_names, sep='-'):
    """
    Create a cluster matrix based on by labelling cells by sample/batch.
    """

    sc = np.array([''] * clusters.shape[0], dtype=object)

    if isinstance(sample_names, dict):
        for sample, id in sample_names.items():
            sc[np.where(clusters.index.str.contains(id))[0]] = sample
    else:
        c = 1

        for s in sample_names:
            id = f'{sep}{c}'
            print(id)
            print(np.where(clusters.index.str.contains(id))[0])
            sc[np.where(clusters.index.str.contains(id))[0]] = s
            c += 1

    df = pd.DataFrame(sc, index=clusters.index, columns=['Cluster'])

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
