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




def cluster_outlines(d,
                     clusters,
                     sdmax=10,
                     ax=None):
    for cluster in clusters['Cluster'].unique():
        cluster_outline(d,
                        cluster,
                        clusters,
                        sdmax=sdmax,
                        ax=ax)


def cluster_outline(d,
                    cluster,
                    clusters,
                    sdmax=10,
                    ax=None):
    """
    Draw outline around cluster, attempt to use bulk
    of cells so that outliers do not affect it too much

    Args:
        tsne (_type_): _description_
        cluster (_type_): _description_
        clusters (_type_): _description_
        sdmax (float, optional): _description_. Defaults to 0.5.
        ax (_type_, optional): _description_. Defaults to None.
    """
    barcodes = clusters[clusters['Cluster'] == cluster].index.values

    d_cluster = d[d.index.isin(barcodes)]

    centroid = (d_cluster.sum(axis=0) / d_cluster.shape[0]).values

    x = d_cluster.iloc[:, 0].values  # data['{}-{}'.format(t, d1)][idx]
    y = d_cluster.iloc[:, 1].values  # data['{}-{}'.format(t, d2)][idx]

    d = np.array([distance.euclidean(centroid, (a, b)) for a, b in zip(x, y)])

    da = abs(d)

    q1 = np.percentile(da, 25)
    q2 = np.percentile(da, 75)
    iqr = q2 - q1

    outlier = q2 + iqr
    idx = np.where(da < outlier)[0]  # (d > x1) & (d < x2))[0]
    x = x[idx]
    y = y[idx]

    sd = d.std()
    m = d.mean()

    z = (d - m) / sd

    za = abs(z)

    iza = 1 / za ** 2

    if (cluster == 8 or cluster == 2):
        print('sdsdf', cluster, za.max(), np.median(za))

    # find all points within 1 sd of centroid

    idx = np.where(abs(z) < sdmax)[0]  # (d > x1) & (d < x2))[0]
    #x = x[idx]
    #y = y[idx]

    points = np.array([[p1, p2] for p1, p2 in zip(x, y)])
    hull = ConvexHull(points)

    xp = points[hull.vertices, 0]
    yp = points[hull.vertices, 1]

    xp = np.append(xp, xp[0])
    yp = np.append(yp, yp[0])

    ax.plot(xp, yp, 'k-')
    cc = plt.Circle((centroid[0], centroid[1]), 0.2)
    ax.add_artist(cc)
    #ax.plot(points[hull.vertices[0], 0], points[hull.vertices[[0, -1]], 1])

    #points = np.array([[x, y] for x, y in zip(x1, y1)])
    #hull = ConvexHull(points)
    #ax.plot(points[hull.vertices,0], points[hull.vertices,1])

    #zi = griddata((x, y), avg1, (xi, yi))

    #ax.contour(xi, yi, z, levels=1)
