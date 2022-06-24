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
from .outline import cluster_outlines


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
                      outline=False,
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

    if outline:
        cluster_outlines(d, clusters, ax=ax)

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


def create_cluster_plots(pca, labels, name, marker='o', s=MARKER_SIZE):
    for i in range(0, pca.shape[1]):
        for j in range(i + 1, pca.shape[1]):
            create_cluster_plot(pca, labels, name, pc1=(
                i + 1), pc2=(j + 1), marker=marker, s=s)
