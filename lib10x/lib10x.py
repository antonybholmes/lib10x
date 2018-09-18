#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:51:15 2018

@author: antony
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import numpy as np
import scipy.sparse as sp_sparse
import tables
import pandas as pd
import sys
sys.path.append('/ifs/scratch/cancer/Lab_RDF/abh2138/scripts/python/')
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import os
import phenograph
import libplot
import libcluster
import libtsne
import libsparse


MARKER_SIZE = 10

SUBPLOT_SIZE = 4
BACKGROUND_SAMPLE_COLOR = (0.92, 0.92, 0.92, libplot.ALPHA)


np.random.seed(0)

GeneBCMatrix = collections.namedtuple('GeneBCMatrix', ['gene_ids', 'gene_names', 'barcodes', 'matrix'])

def decode(items):
  return np.array([x.decode('utf-8') for x in items])

def get_matrix_from_h5(filename, genome):
    with tables.open_file(filename, 'r') as f:
        try:
            dsets = {}
            
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()
                
            matrix = sp_sparse.csc_matrix((dsets['data'], dsets['indices'], dsets['indptr']), shape=dsets['shape'])
            return GeneBCMatrix(decode(dsets['genes']), decode(dsets['gene_names']), decode(dsets['barcodes']), matrix)
        except tables.NoSuchNodeError:
            raise Exception("Genome %s does not exist in this file." % genome)
        except KeyError:
            raise Exception("File is missing one or more required datasets.")

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


def create_tsne_cluster_plots(pca, labels, name, marker='o', s=MARKER_SIZE):
    for i in range(0, pca.shape[1]):
        for j in range(i + 1, pca.shape[1]):
            create_tsne_cluster_plot(pca, labels, name, pc1 = (i + 1), pc2 = (j + 1), marker=marker, s=s)
            
            
def pca_base_plots(pca, clusters, n=10, marker='o', s=MARKER_SIZE):
    rows = libplot.grid_size(n)
    
    w = 4 * rows
    
    fig = libplot.new_base_fig(w=w, h=w)
    
    si = 1
    
    for i in range(0, n):
        for j in range(i + 1, n):
            ax = libplot.new_ax(fig, subplot=(rows, rows, si))
            
            pca_plot_base(pca, clusters, pc1 = (i + 1), pc2 = (j + 1), marker=marker, s=s, ax=ax)
            
            si += 1
            
    return fig


def pca_plot_base(pca, clusters, pc1=1, pc2=2, marker='o', s=MARKER_SIZE, fig=None, ax=None):
    colors = libcluster.colors()
    
    if ax is None:
        fig, ax = libplot.new_fig()
  
    ids = list(sorted(set(clusters['Cluster'])))
  
    for i in range(0, len(ids)):
        l = ids[i]
        
        #print('Label {}'.format(l))
        indices = np.where(clusters['Cluster'] == l)[0]
        
        n = len(indices)
        
        label = 'C{} ({:,})'.format(l, n)
    
        df2 = pca.iloc[indices,]
          
        ax.scatter(df2['PC-{}'.format(pc1)], df2['PC-{}'.format(pc2)], color=colors[i], edgecolor=colors[i], s=s, marker=marker, alpha=libplot.ALPHA, label=label)
           
    return fig, ax


def pca_plot(pca, clusters, pc1=1, pc2=2, marker='o', s=MARKER_SIZE, fig=None, ax=None):
    fig, ax = pca_plot_base(pca, clusters, pc1=pc1, pc2=pc2, marker=marker, s=s, fig=fig, ax=ax)
    
    #libtsne.tsne_legend(ax, labels, colors)
    libcluster.format_simple_axes(ax, title="PC")
    libcluster.format_legend(ax, cols=6, markerscale=2)
    
    return fig, ax
    
    
def create_pca_plot(pca, clusters, name, pc1=1, pc2=2, marker='o', s=MARKER_SIZE, fig=None, ax=None):
    out = 'pca_{}_pc{}_vs_pc{}.pdf'.format(name, pc1, pc2)
   

    fig, ax = pca_plot(pca, clusters, pc1=pc1, pc2=pc2, marker=marker, s=s, fig=fig, ax=ax)
    
    libplot.savefig(fig, out, pad=2)
    plt.close(fig)


def base_tsne_cluster_plot(tsne, clusters, marker='o', s=libplot.MARKER_SIZE, c=None, fig=None, ax=None):
    """
    Create a tsne plot without the formatting
    """
    
    if ax is None:
        fig, ax = libplot.new_fig()
    
    libcluster.scatter_clusters(tsne['TSNE-1'], tsne['TSNE-2'], clusters, c=c, marker=marker, s=s, ax=ax)
    
    libcluster.format_axes(ax)
    libcluster.format_legend(ax, cols=6, markerscale=2)
    
    return fig, ax


def tsne_cluster_plot(tsne, clusters, marker='o', s=libplot.MARKER_SIZE, c=None, fig=None, ax=None):
    fig, ax = base_tsne_cluster_plot(tsne, clusters, marker=marker, c=c, s=s, fig=fig, ax=ax)
    
    #libtsne.tsne_legend(ax, labels, colors)
    libcluster.format_simple_axes(ax, title="t-SNE")
    #libcluster.format_legend(ax, cols=6, markerscale=2)
    
    return fig, ax


def create_tsne_cluster_plot(tsne_results, clusters, name, marker='o', s=libplot.MARKER_SIZE, ax=None):
    out = libtsne.get_tsne_plot_name(name)
      
    fig, ax = tsne_cluster_plot(tsne_results, clusters, marker=marker, s=s)
    
    libplot.savefig(fig, out, pad=2)
    plt.close(fig)
    
    return fig, ax


def base_tsne_plot(tsne, marker='o', s=libplot.MARKER_SIZE, c='red', label=None, fig=None, ax=None):
    """
    Create a tsne plot without the formatting
    """
    
    if ax is None:
        fig, ax = libplot.new_fig()
        
    libplot.scatter(tsne['TSNE-1'], tsne['TSNE-2'], c=c, marker=marker, label=label, s=s, ax=ax)
    
    return fig, ax


def tsne_plot(tsne, marker='o', s=libplot.MARKER_SIZE, c='red', label=None, fig=None, ax=None):
    fig, ax = base_tsne_plot(tsne, marker=marker, c=c, s=s, label=label, fig=fig, ax=ax)
    
    #libtsne.tsne_legend(ax, labels, colors)
    libcluster.format_simple_axes(ax, title="t-SNE")
    libcluster.format_legend(ax, cols=6, markerscale=2)
    
    return fig, ax



def base_expr_plot(data, exp, d1=1, d2=2, t='TSNE', x1=None, x2=None, cmap=plt.cm.plasma, marker='o', edgecolors='none', s=MARKER_SIZE, alpha=libplot.ALPHA, fig=None, ax=None, norm=None): #plt.cm.plasma):
    """
    Basic function for creating an expression plot for t-sne/2D space
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
    t : str, optional
        Indicate datatype, e.g. 'TSNE', or 'PCA'
    fig : matplotlib figure, optional
        Supply a figure object on which to render the plot, otherwise a new
        one is created.
    ax : matplotlib ax, optional
        Supply an axis object on which to render the plot, otherwise a new
        one is created.
        
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
      fig, ax = libplot.new_fig()
    
    # Sort by expression level
    idx = np.argsort(exp)
    
    x = data['{}-{}'.format(t, d1)][idx]
    y = data['{}-{}'.format(t, d2)][idx]
    e = exp[idx]
    
    ax.scatter(x, y, c=e, s=s, marker=marker, alpha=alpha, cmap=cmap, norm=norm, edgecolors=edgecolors)
    
    libcluster.format_axes(ax, title=t)
    
    return fig, ax


def expr_plot(data, exp, t='TSNE', d1=1, d2=2, x1=None, x2=None, cmap=plt.cm.plasma, marker='o', s=MARKER_SIZE, alpha=libplot.ALPHA, fig=None, ax=None, norm=None): #plt.cm.plasma):
    """
    Creates a basic expression plot and adds a color bar
    """
    
    is_first = False
    
    if ax is None:
      fig, ax = libplot.new_fig()
      is_first = True
    
    base_expr_plot(data, exp, d1=d1, d2=d2, t=t, s=s, marker=marker, alpha=alpha, cmap=cmap, norm=norm, ax=ax)
    
    if is_first:
        libplot.add_colorbar(fig, cmap)
        #libcluster.format_simple_axes(ax, title=t)
  
    return fig, ax


def tsne_expr_plot(data, exp, d1=1, d2=2, x1=None, x2=None, cmap=plt.cm.plasma, marker='o', s=MARKER_SIZE, alpha=libplot.ALPHA, fig=None, ax=None, norm=None): #plt.cm.plasma):
    """
    Creates a basic t-sne expression plot and adds a color bar
    """
    
    fig, ax = expr_plot(data, exp, t='TSNE', d1=d1, d2=d2, x1=x1, x2=x2, cmap=cmap, marker=marker, s=s, alpha=alpha, fig=fig, ax=ax, norm=norm)
    
    return fig, ax


def create_tsne_expr_plot(data, exp, name, d1=1, d2=2, x1=None, x2=None, cmap=plt.cm.plasma, marker='o', s=MARKER_SIZE, alpha=libplot.ALPHA, fig=None, ax=None, norm=None): #plt.cm.plasma):
    """
    Creates and saves a presentation tsne plot
    """
    
    out = 'tsne_expr_{}_t{}_vs_t{}.pdf'.format(name, 1, 2)
    

    fig, ax = tsne_expr_plot(data, exp, d1=d1, d2=d2, x1=x1, x2=x2, cmap=cmap, marker=marker, s=s, alpha=alpha, fig=fig, ax=ax, norm=norm)
    
    libcluster.format_simple_axes(ax)
    
    libplot.savefig(fig, out)
    plt.close(fig)
  
    return fig, ax


def base_pca_expr_plot(data, exp, d1=1, d2=2, x1=None, x2=None, cmap=plt.cm.plasma, marker='o', s=MARKER_SIZE, alpha=libplot.ALPHA, fig=None, ax=None, norm=None): #plt.cm.plasma):
    fig, ax = base_expr_plot(data, exp, t='PC', d1=d1, d2=d2, x1=x1, x2=x2, cmap=cmap, marker=marker, s=s, fig=fig, alpha=alpha, ax=ax, norm=norm)
    
    return fig, ax


def pca_expr_plot(data, expr, name, d1=1, d2=2, x1=None, x2=None, cmap=plt.cm.plasma, marker='o', s=MARKER_SIZE, alpha=libplot.ALPHA, fig=None, ax=None, norm=None): #plt.cm.plasma):
    out = 'pca_expr_{}_t{}_vs_t{}.pdf'.format(name, 1, 2)
      
    fig, ax = base_pca_expr_plot(data, expr, d1=d1, d2=d2, x1=x1, x2=x2, cmap=cmap, marker=marker, s=s, alpha=alpha, fig=fig, ax=ax, norm=norm)
    
    libplot.savefig(fig, out)
    plt.close(fig)
  
    return fig, ax


def expr_grid_size(x):
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
    
    w = SUBPLOT_SIZE * cols
    
    rows = int(l / cols) + 1
    
    if l % cols == 0:
        # Assume we will add a row for a color bar
        rows += 1
        
    h = SUBPLOT_SIZE * rows
    
    return w, h, rows, cols

    
def tsne_gene_expr_grid(data, tsne, genes):
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
    
    w, h, rows, cols = expr_grid_size(genes)
    
    cmap = plt.cm.plasma
    
    fig = libplot.new_base_fig(w=w, h=h)
    
    for i in range(0, len(genes)):
        gene = genes[i]
        
        if type(gene) == list:
            exp = np.zeros(data.shape[1])
            
            for g in gene:
                exp += data.loc[data.index.str.endswith(g), :].iloc[0,:].values
            
            #exp /= len(gene)
        else:
            if isinstance(data.index, np.ndarray):
                idx = [i for i, v in enumerate(data.index) if v.endswith(gene)]
                
                exp = data[idx, :].toarray()[0]
            else:
                exp = data.loc[data.index.str.endswith(gene), :].iloc[0,:].values
        
        ax = libplot.new_ax(fig, rows, cols, i + 1)
        
        
        
        tsne_expr_plot(tsne, exp, ax=ax)
        
        #if i == 0:
        #    libcluster.format_axes(ax)
        #else:
            
        libplot.invisible_axes(ax)
            
        ax.set_title(gene)
        
    libplot.add_colorbar(fig, cmap)    

    return fig



def tsne_cluster_grid(tsne, clusters, colors=None):
    """
    Plot each cluster separately to highlight where the samples are
    
    Parameters
    ----------
    tsne : Pandas dataframe
        Cells x tsne tsne data. Columns should be labeled 'TSNE-1', 'TSNE-2' etc
    clusters : DataFrame
        Clusters in 
        
    Returns
    -------
    fig : Matplotlib figure
        A new Matplotlib figure used to make the plot
    """
    
    
    ids = list(sorted(set(clusters['Cluster'])))
    
    rows = int(np.ceil(np.sqrt(len(ids))))
    
    w = SUBPLOT_SIZE * rows
    
    fig = libplot.new_base_fig(w=w, h=w)
    
    if colors is None:
        colors = libcluster.colors()
    
    for i in range(0, len(ids)):
        c = ids[i]
        
        #print('Label {}'.format(l))
        idx1 = np.where(clusters['Cluster'] == c)[0]
        idx2 = np.where(clusters['Cluster'] != c)[0]
        
        
        ax = libplot.new_ax(fig, rows, rows, i + 1)
        
        x = tsne.iloc[idx2, 0]
        y = tsne.iloc[idx2, 1]
        
        libplot.scatter(x, y, c=BACKGROUND_SAMPLE_COLOR, ax=ax)
        
        x = tsne.iloc[idx1, 0]
        y = tsne.iloc[idx1, 1]
        
        libplot.scatter(x, y, c=colors[i], ax=ax)
    
        ax.set_title('C{} ({})'.format(c, len(idx1)), color=colors[i])
        libplot.invisible_axes(ax)
            
    return fig

def create_tsne_cluster_grid(tsne, clusters, name, colors=None):
    fig = tsne_cluster_grid(tsne, clusters, colors)
    libplot.savefig(fig, 'tsne_{}_separate_clusters.pdf'.format(name))



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
  
  labels = data #.tolist()
  
  return cluster_map, labels



def umi_norm(data):
    # each column is a cell
    reads_per_bc = data.sum(axis=0)
    median_reads_per_bc = np.median(reads_per_bc)
    scaling_factors = median_reads_per_bc / reads_per_bc
    
    print(data.shape, len(scaling_factors))
    scaled = data.multiply(scaling_factors) #, axis=1)
    
    return scaled


def umi_norm_log2(data):
    if isinstance(data, libsparse.SparseDataFrame):
        return data.log2(add=1)
    else:
        return (umi_norm(data) + 1).apply(np.log2)


def umi_norm_log2_scale(data):
    d = umi_norm_log2(data).T
    
    if isinstance(data, libsparse.SparseDataFrame):
        sd = StandardScaler(with_mean=False).fit_transform(d.matrix)
        
        return libsparse.SparseDataFrame(sd.T, index=data.index, columns=data.columns)
    else:
        sd = StandardScaler().fit_transform(d)
    
        return pd.DataFrame(sd.T, index=data.index, columns=data.columns)


def read_clusters(file):
  print('Reading clusters from {}...'.format(file))
  
  return pd.read_csv(file, sep='\t', header=0, index_col=0)


def silhouette(tsne, tsne_umi_log2, clusters, name):
    # measure cluster worth
    x1=silhouette_samples(tsne, clusters.iloc[:,0].tolist(), metric='euclidean')
    x2=silhouette_samples(tsne_umi_log2, clusters.iloc[:,0].tolist(), metric='euclidean')
    
    fig, ax = libplot.newfig(w=9, h=7, subplot=211)
    df = pd.DataFrame({'Silhouette Score':x1, 'Cluster':clusters.iloc[:,0].tolist(), 'Label':np.repeat('tsne-10x', len(x1))})
    libplot.boxplot(df, 'Cluster', 'Silhouette Score', colors=libcluster.colors(), ax=ax)
    ax.set_ylim([-1, 1])
    ax.set_title('tsne-10x')
    #libplot.savefig(fig, 'RK10001_10003_clust-phen_silhouette.pdf')
    
    ax = fig.add_subplot(212) #libplot.newfig(w=9)
    df2 = pd.DataFrame({'Silhouette Score':x2, 'Cluster':clusters.iloc[:,0].tolist(), 'Label':np.repeat('tsne-ah', len(x2))})
    libplot.boxplot(df2, 'Cluster', 'Silhouette Score', colors=libcluster.colors(), ax=ax)
    ax.set_ylim([-1, 1])
    ax.set_title('tsne-ah')
    libplot.savefig(fig, '{}_silhouette.pdf'.format(name))
    

def node_color_from_cluster(clusters):
    colors = libcluster.colors()
    
    return [colors[clusters['Cluster'][i] - 1] for i in range(0, clusters.shape[0])]

#def network(tsne, clusters, name, k=5):
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
    G=nx.from_numpy_matrix(A)
    pos=nx.spring_layout(G)
    
    fig, ax = libplot.newfig(w=8, h=8)
    node_color = libcluster.colors()[0:c.shape[0]] #list(range(0, c.shape[0]))
    cmap=libcluster.colormap()
    
    labels ={}
    
    for i in range(0, c.shape[0]):
        labels[i] = i + 1

    #nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, node_size=200, node_color=node_color, vmax=(c.shape[0] - 1), cmap=libcluster.colormap())
    nx.draw_networkx(G, with_labels=True, labels=labels, ax=ax, node_size=800, node_color=node_color, font_color='white', font_family='Arial')
    
    libplot.format_axes(ax)
    libplot.savefig(fig, '{}_centroid_network.pdf'.format(name))


def centroids(tsne, clusters):
    cids = list(sorted(set(clusters['Cluster'].tolist())))
    
    ret = np.zeros((len(cids), 2))
    
    for i in range(0, len(cids)):
        c = cids[i]
        x = tsne.iloc[np.where(clusters['Cluster'] == c)[0],:]
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
        ids1 = np.where(a1[i,:] > 0)[0]
        ids2 = np.where(a2[i,:] > 0)[0]
        ids3 = np.intersect1d(ids1, ids2)
        o = len(ids3) / 5 * 100
        overlaps.append(o)
        
    df = pd.DataFrame({'Cluster':list(range(1, c1.shape[0] + 1)), 'Overlap %':overlaps})
    df.set_index('Cluster', inplace=True)
    df.to_csv('{}_cluster_overlaps.txt'.format(name), sep='\t')
    
    


