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
import libtsne
import seaborn as sns
from libsparse.libsparse import SparseDataFrame
from lib10x.sample import *
from scipy.spatial import ConvexHull
from PIL import Image, ImageFilter

from scipy.stats import binned_statistic

from .outline import cluster_outline
from .color import get_colors

import imagelib

from .constants import *

from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import sys
import matplotlib.colors as mcolors
import libplot
from libsparse.libsparse import SparseDataFrame


def pca(data, n=50, exclude=[], mode='random'):
    """
    Setup the PCA on a data set
    """
  
    # remove rows containing all zeros
    #data = remove_empty_rows(data) # data[(data.T != 0).any()]
  
    print('New size {}'.format(data.shape))
  
    #if tpmmode:
    #  print('Converting to tpm...')
    #  data = tpm(data)
  
    # In log mode, convert to log2 tpm
    #if logmode:
    #  print('Converting to log2...')
    #  data = log2(data)
    
    
    #data = data.values #as_matrix()

      # Scale between 0 and 1
      #data_std = StandardScaler().fit_transform(datat)
  
    data = data.T #np.transpose(data)
  
    print('tranpose {}'.format(data.shape))
  
    # Perform PCA
    
    print('hmmw', n, data.shape[0])
    
    n = min(data.shape[0], n)
    
    
  
    if isinstance(data, SparseDataFrame):
        print('PCA sparse mode')
        pca = TruncatedSVD(n_components=n, random_state=PCA_RANDOM_STATE)
    elif mode == 'full':
        print('PCA full mode')
        pca = PCA(n_components=n, svd_solver='full', random_state=PCA_RANDOM_STATE)
    else:
        print('PCA random mode')
        # Use the default random, faster solver
        pca = PCA(n_components=n, random_state=PCA_RANDOM_STATE)
    
    if isinstance(data, SparseDataFrame):
        pca_results = pca.fit_transform(data.matrix) #libsparse.SparseDataFrame(pca.fit_transform(data.matrix), data.index, data.columns)
    else:
        pca_results = pca.fit_transform(data) #data_std) #datat)
  
    if len(exclude) > 0:
        # filter
        pca_results = pca_results[:, np.where(np.in1d(list(range(0, n)), exclude, invert=True))[0]]
  
    print(pca_results.shape)
  
    return pca, pca_results


def lighter(color, percent):
  '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
  color = np.array(color)
  white = np.array([255, 255, 255])
  vector = white-color
  
  return color + vector * percent
    

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
      U, s, Vt = np.linalg.svd(covariance)
      angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
      width, height = 2 * np.sqrt(s)
    else:
      angle = 0
      width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 3):
      ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

      
                         
        
def plot_gmm(gmm, X, colors, colormap, label=True, ax=None):
  ax = ax or plt.gca()
  labels = gmm.fit(X).predict(X)
  
  if label:
      #ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
      ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=colormap, s=MARKER_SIZE, alpha=ALPHA, zorder=2, norm=mcolors.Normalize(vmin=0, vmax=7))
  else:
      ax.scatter(X[:, 0], X[:, 1], alpha=alpha, s=MARKER_SIZE, zorder=2)
  
  #ax.axis('equal')
  
  w_factor = 0.4 / gmm.weights_.max()
  
  for i in range(0, len(gmm.means_)):
    pos = gmm.means_[i]
    covar = gmm.covariances_[i]
    w = gmm.weights_[i]
    color = colors[i]
    
    #w in zip(gmm.means_, gmm.covariances_, gmm.weights_
    #for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    
    sys.stderr.write("w " + str(w) + "\n")
    draw_ellipse(pos, covar, facecolor=color, alpha=w * w_factor)

  return labels


def plot_cluster_ellipses(gmm, X, colors, ax=None):
  ax = ax or plt.gca()
  labels = gmm.fit(X).predict(X)
  
  w_factor = 0.25 / gmm.weights_.max()
  
  for i in range(0, len(gmm.means_)):
    pos = gmm.means_[i]
    covar = gmm.covariances_[i]
    w = gmm.weights_[i]
    color = 'dimgray' #colors[i]
    
    draw_ellipse(pos, covar, edgecolor=color, facecolor='none', alpha=0.5) #(w * w_factor))

  return labels


def draw_outline(x, y, ax, color):
  points = np.transpose([x, y])
  
  print("points", str(len(points)))
  
  hull = ConvexHull(points)
  
   # close the polygon shape
  vertices = np.append(hull.vertices, hull.vertices[0])
  
  points = points[vertices]
  
  x = points[:, 0]
  y = points[:, 1]
  
  ax.plot(x, y, '--', lw=1, color=color, alpha=0.5)
  
  return x, y
  
  #t = np.arange(len(x))
  #ti = np.linspace(0, t.max(), 200)

  #xi = interp1d(t, x, kind='cubic')(ti)
  #yi = interp1d(t, y, kind='cubic')(ti)
  
  #ax.plot(xi, yi, '--', lw=1, color=color, alpha=0.4)
  
  #return xi, yi
  
 
def fill_outline(x, y, ax, color):
  x, y = draw_outline(x, y, ax, color)
  
  plt.fill(x, y, color=color, alpha=0.15)
  
  
def plot_cluster_outlines(gmm, X, colors, colormap, lz_indices, dz_indices, ax=None):
  ax = ax or plt.gca()
  labels = gmm.fit(X).predict(X)
  
  #
  # LZ
  #
  
  # red
  indices = np.intersect1d(lz_indices, label_indices(labels, 0))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    draw_outline(x1, x2, ax, 'red')
   
  # green
  indices = np.intersect1d(lz_indices, label_indices(labels, 1))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    draw_outline(x1, x2, ax, 'green')
    
  # blue
  indices = np.intersect1d(lz_indices, label_indices(labels, 2))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    draw_outline(x1, x2, ax, 'blue')
  
  #
  # DZ
  #
  
  # red
  indices = np.intersect1d(dz_indices, label_indices(labels, 0))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    draw_outline(x1, x2, ax, 'red')
    
  # green
  indices = np.intersect1d(dz_indices, label_indices(labels, 1))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    draw_outline(x1, x2, ax, 'green')
    
  # blue
  indices = np.intersect1d(dz_indices, label_indices(labels, 2))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    draw_outline(x1, x2, ax, 'blue')

  return labels


def plot_gmm2(gmm, X, colors, colormap, lz_indices, dz_indices, ax=None):
  ax = ax or plt.gca()
  labels = gmm.fit(X).predict(X)
  
  w_factor = 0.25 / gmm.weights_.max()
  
  for i in range(0, len(gmm.means_)):
    pos = gmm.means_[i]
    covar = gmm.covariances_[i]
    w = gmm.weights_[i]
    color = colors[i]
    
    draw_ellipse(pos, covar, facecolor=color, alpha=(w * w_factor))
  
  #
  # LZ
  #
  
  # red
  indices = np.intersect1d(lz_indices, label_indices(labels, 0))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='tomato', s=MARKER_SIZE, marker='^', edgecolor=EDGE_COLORS[0], alpha=ALPHA, zorder=10)
   
    
  # green
  indices = np.intersect1d(lz_indices, label_indices(labels, 1))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='mediumseagreen', s=MARKER_SIZE, marker='^', edgecolor=EDGE_COLORS[1], alpha=ALPHA, zorder=10)
    
  # blue
  indices = np.intersect1d(lz_indices, label_indices(labels, 2))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='royalblue', s=MARKER_SIZE, marker='^', edgecolor=EDGE_COLORS[2], alpha=ALPHA, zorder=10)
  
  #
  # DZ
  #
  
  # red
  indices = np.intersect1d(dz_indices, label_indices(labels, 0))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='tomato', s=MARKER_SIZE, marker='o', edgecolor=EDGE_COLORS[0], alpha=ALPHA, zorder=10)
    
  # green
  indices = np.intersect1d(dz_indices, label_indices(labels, 1))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='mediumseagreen', s=MARKER_SIZE, marker='o', edgecolor=EDGE_COLORS[1], alpha=ALPHA, zorder=10)
    
  # blue
  indices = np.intersect1d(dz_indices, label_indices(labels, 2))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='royalblue', s=MARKER_SIZE, marker='o', edgecolor=EDGE_COLORS[2], alpha=ALPHA, zorder=10)

  return labels

  

def plot_gmm3(gmm, X, colors, colormap, lz_indices, dz_indices, ax=None):
  ax = ax or plt.gca()
  labels = gmm.fit(X).predict(X)
  
  #
  # Outlines
  #
  
  lz_dz_indices = lz_indices + dz_indices
  
  indices = np.intersect1d(lz_dz_indices, label_indices(labels, 0))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    fill_outline(x1, x2, ax, 'red')
   
  # green
  indices = np.intersect1d(lz_dz_indices, label_indices(labels, 1))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    fill_outline(x1, x2, ax, 'green')
    
  # blue
  indices = np.intersect1d(lz_dz_indices, label_indices(labels, 2))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    fill_outline(x1, x2, ax, 'blue')
  
  
  
  #
  # LZ
  #
  
  # red
  indices = np.intersect1d(lz_indices, label_indices(labels, 0))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='tomato', s=MARKER_SIZE, marker='^', edgecolor=EDGE_COLORS[0], alpha=ALPHA, zorder=10)
   
  # green
  indices = np.intersect1d(lz_indices, label_indices(labels, 1))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='mediumseagreen', s=MARKER_SIZE, marker='^', edgecolor=EDGE_COLORS[1], alpha=ALPHA, zorder=10)
    
  # blue
  indices = np.intersect1d(lz_indices, label_indices(labels, 2))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='royalblue', s=MARKER_SIZE, marker='^', edgecolor=EDGE_COLORS[2], alpha=ALPHA, zorder=10)
  
  #
  # DZ
  #
  
  # red
  indices = np.intersect1d(dz_indices, label_indices(labels, 0))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='tomato', s=MARKER_SIZE, marker='o', edgecolor=EDGE_COLORS[0], alpha=ALPHA, zorder=10)
    
    
  # green
  indices = np.intersect1d(dz_indices, label_indices(labels, 1))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='mediumseagreen', s=MARKER_SIZE, marker='o', edgecolor=EDGE_COLORS[1], alpha=ALPHA, zorder=10)
    
  # blue
  indices = np.intersect1d(dz_indices, label_indices(labels, 2))
  
  if len(indices) > 0:
    x1, x2 = filter_x(X, 0, 1, indices)
    l = np.take(labels, indices)
    ax.scatter(x1, x2, color='royalblue', s=MARKER_SIZE, marker='o', edgecolor=EDGE_COLORS[2], alpha=ALPHA, zorder=10)

  return labels


def plot_louvain(labels, X, colors, lz_indices, dz_indices, ax):
  # How many labels to cycle through (it cannot exceed the number of colors)
  
  labeln = min(len(colors), np.max(labels) + 1)
  
  for l in range(0, labeln):
    li = label_indices(labels, l)
    
    indices = np.intersect1d(lz_indices, li)
  
    if len(indices) > 0:
      x1, x2 = filter_x(X, 0, 1, indices)
      ax.scatter(x1, x2, color=colors[l], edgecolor=colors[l], s=MARKER_SIZE, marker='^', alpha=0.8, zorder=10)
      
    indices = np.intersect1d(dz_indices, li)
  
    if len(indices) > 0:
      x1, x2 = filter_x(X, 0, 1, indices)
      ax.scatter(x1, x2, color=colors[l], edgecolor=colors[l], s=MARKER_SIZE, marker='o', alpha=0.8, zorder=10)
      
  
def plot_louvain_lz_dz(labels, X, colors, lz_indices, dz_indices, ax):
  # How many labels to cycle through (it cannot exceed the number of colors)
  print(len(colors))
  print(np.max(labels))
  labeln = min(len(colors), np.max(labels) + 1)
  
  for l in range(0, labeln):
    li = label_indices(labels, l)
    
    indices = np.intersect1d(lz_indices, li)
  
    if len(indices) > 0:
      x1, x2 = filter_x(X, 0, 1, indices)
      ax.scatter(x1, x2, color='white', edgecolor='black', s=MARKER_SIZE, marker='^', alpha=0.7, zorder=10)
      
    indices = np.intersect1d(dz_indices, li)
  
    if len(indices) > 0:
      x1, x2 = filter_x(X, 0, 1, indices)
      ax.scatter(x1, x2, color='white', edgecolor='black', s=MARKER_SIZE, marker='o', alpha=0.7, zorder=10)
      
  
def label_indices(labels, l):
  return np.where(labels == l)[0]
  
  
#def make_colormap():
#  colors = []
#  colors.append('red')
#  colors.append('green')
#  colors.append('blue')
#  colors.append('orange')
#  colors.append('violet')
#  colors.append('gold')
#  colors.append('gray')
#  colors.append('black')
#  
#  return [colors, mcolors.ListedColormap(colors, name='gmm')]




def write_tables(data, labels, colors, clusters, prefix):
  ulabels = set(labels)

  for label in sorted(ulabels):
    indices = np.where(labels == label)
    
    d = data[data.columns[indices]]
    
    file = prefix + "_c" + str(clusters) + "_" + colors[label] + ".txt"
    
    print(file)
    
    d.to_csv(file, sep="\t", header=True, index=True)
    
    #print indices


def write_groups(data, labels, colors, clusters, prefix):
  ulabels = set(labels)
  
  dir = 'c' + str(clusters)
  
  if not os.path.exists(dir):
    os.makedirs(dir)

  for label in sorted(ulabels):
    indices = np.where(labels == label)
    
    d = data[data.columns[indices]]
    
    file = dir + '/' + prefix + "_c" + str(clusters) + "_" + colors[label] + ".txt"
    
    print(file)
    
    #d.to_csv(file, sep="\t", header=True, index=True)
    
    f = open(file, 'w')
    
    f.write("Sample ID\n")
    
    for c in data.columns.values[indices]:
      f.write(c + "\n")
    
    f.close()
    
    
def find_indices(df, search):
  return np.where(df.columns.str.contains(search))[0].tolist() # pca = pca.T [df.columns.get_loc(c) for c in df.filter(like=search).columns]


#def format_axes(ax):
#  ax.spines['right'].set_visible(False)
#  ax.spines['top'].set_visible(False)
#  ax.spines['bottom'].set_color('dimgray')
#  ax.spines['left'].set_color('dimgray')
#  ax.minorticks_on()
#  ax.get_yaxis().set_tick_params(which='both', direction='in')
#  ax.get_xaxis().set_tick_params(which='both', direction='in')
  

def format_legend(ax, cols=6, markerscale=None):
  ax.legend(bbox_to_anchor=[0, 0.95], loc='lower left', ncol=cols, frameon=False, fontsize='small', markerscale=markerscale, handlelength=1, columnspacing=0.5)

def format_legend_2(ax):
  ax.legend(bbox_to_anchor=[0.95, 0.95], loc='lower left', ncol=1, frameon=False, fontsize='small', handlelength=1, columnspacing=1)


def make_figure(w=8, h=8):
  """
  Make a figure of uniform size
  """
  fig = plt.figure(figsize=(w, h))
  ax = fig.add_subplot(1, 1, 1)
  
  return [fig, ax]
  

  
def save_plot(fig, out, pad=3):
  fig.tight_layout(pad=pad) #rect=[o, o, w, w])

  plt.savefig(out, dpi=600)


def filter_x(x, c1, c2, indices):
  x1 = np.take(x[:, c1], indices)
  x2 = np.take(x[:, c2], indices)
  
  return x1, x2


def filter_log2(data):
  datat = data[(data.T != 0).any()]
  
  # Transpose for PCA
  datat = datat.transpose().as_matrix()
  # Add 1 for log
  datat += 1
  datat = np.log2(datat)
  
  return datat


def log2(data):
  #data = data.as_matrix()
  # Add 1 for log
  #data += 1
  #data = np.log2(data)
  
  return (data + 1).apply(np.log2)


def tpm(data):
  return data / data.sum(axis=0) * 1000000



def remove_empty_cols(data):
    if isinstance(data, SparseDataFrame):
        return data.remove_empty_cols()
    else:
        if isinstance(data, ndarray):
            data = pd.DataFrame(data)
            
        #return data.loc[(data != 0).any(1)]
        ret = data.loc[:, data.sum(axis=0) != 0]
        
        return ret
    
def remove_empty_rows(data):
    return remove_empty_cols(data.T).T
    
def remove_empty_cells(data):
    return remove_empty_cols(remove_empty_rows(data))
    
  
def format_axes(ax, title="t-SNE", d1=1, d2=2, subtitle1="", subtitle2=""):
    if subtitle1 != "":
        ax.set_xlabel('{} {} ({})'.format(title, d1, subtitle1))
    else:
        ax.set_xlabel('{} {}'.format(title, d1))
  
    if subtitle2 != "":
        ax.set_ylabel('{} {} ({})'.format(title, d2, subtitle2))
    else:
        ax.set_ylabel('{} {}'.format(title, d2))
  
    
def format_simple_axes(ax, title="t-SNE", d1=1, d2=2, subtitle1="", subtitle2=""):
  libplot.invisible_axes(ax)
  
  ax.annotate('',
    xy=(40, 0),  # theta, radius
    xytext=(-2, 0),
    xycoords='axes pixels',
    textcoords='axes pixels',
    arrowprops=dict(arrowstyle='->', facecolor='black'), zorder=1000)
  
  
  ax.annotate('',
    xy=(0, 40),  # theta, radius
    xytext=(0, -2),
    xycoords='axes pixels',
    textcoords='axes pixels',
    arrowprops=dict(arrowstyle='->', facecolor='black'), zorder=1000)
  
  if subtitle1 != "":
    ax.text(0, -0.04, '{} {} ({})'.format(title, d1, subtitle1), transform=ax.transAxes)
  else:
    ax.text(0, -0.04, '{} {}'.format(title, d1), transform=ax.transAxes)
  
  if subtitle2 != "":
    ax.text(-0.04, 0, '{} {} ({})'.format(title, d2, subtitle2), va='bottom', transform=ax.transAxes, rotation=90)
  else:
    ax.text(-0.04, 0, '{} {}'.format(title, d2), va='bottom', transform=ax.transAxes, rotation=90)




    
    
def write_group_exp(data, labels, colors, prefix="tsne"):
  ulabels = set(labels)
  
  for label in sorted(ulabels):
    indices = np.where(labels == label)[0]
    
    file = '{}_exp_{}.txt'.format(prefix, label + 1)
    
    print(indices)
    print(data.shape)
    
    d = data.take(indices, axis=1)
    
    print(file)
    
    d.to_csv(file, sep="\t", header=True, index=True)
    
    #f = open(file, 'w')
    
    #f.write("Sample ID\n")
    
    #for c in data.columns.values[indices]:
    #  f.write(c + "\n")
    
    #f.close()
  
  
def write_tsne_groups(data, labels, name):
  ulabels = set(labels)
  
  for label in sorted(ulabels):
    indices = np.where(labels == label)
    
    file = 'tsne_{}_{}.txt'.format(name, label + 1)
    
    #d.to_csv(file, sep="\t", header=True, index=True)
    
    f = open(file, 'w')
    
    f.write("Sample ID\n")
    
    for c in data.columns.values[indices]:
      f.write(c + "\n")
    
    f.close()
    

def shannon_diversity_mode(labels, indices_map):
  """
  Diversity in LZ
  """
  
  label_ids = set(labels)
  
  label_indices_map = collections.defaultdict(list)
  
  
  
  # indices per label
  for l in label_ids:
    indices = np.where(labels == l)[0]
    label_indices_map[l] = indices
  
  score_map = collections.defaultdict(float)
  
  lz_map = collections.defaultdict(lambda: collections.defaultdict(int))
  
  for l in label_ids:
    count_map = collections.defaultdict(int)
    
    label_indices = label_indices_map[l]
    
    n = len(label_indices)
    
    for p in indices_map:
      for mode in indices_map[p]:
        if mode == 'all':
          continue
        
        indices = indices_map[p][mode]
      
        # intersect to find proportions
        overlap = np.intersect1d(label_indices, indices)
        
        c = len(overlap)
        
        count_map[mode] += c
        
        lz_map[l][mode] += c
  
    h = 0
  
    for mode in count_map:
      d = count_map[mode] / n
      
      if d > 0:
        h += d * math.log(d)

    h *= -1
    
    score_map[l] = h
    
  ret = np.zeros(len(labels))
  
  for l in label_ids:
    indices = label_indices_map[l]
    ret[indices] = score_map[l]
  
  return ret, score_map, lz_map


def shannon_diversity(labels, indices_map, name):
  
  
  label_ids = set(labels)
  
  label_indices_map = collections.defaultdict(list)
  
  #
  # Diversity between patients
  #
  
  score_map = collections.defaultdict(float)
  
  patient_map = collections.defaultdict(lambda: collections.defaultdict(int))
  
  # indices per label
  for l in label_ids:
    indices = np.where(labels == l)[0]
    label_indices_map[l] = indices
  
  for l in label_ids:
    count_map = collections.defaultdict(int)
    
    label_indices = label_indices_map[l]
    
    n = len(label_indices)
    
    for p in indices_map:
      indices = indices_map[p]['all']
      
      # intersect to find proportions
      overlap = np.intersect1d(label_indices, indices)
      
      size = len(overlap)
      
      count_map[p] += size
      
      patient_map[l][p] += size
  
    h = 0
  
    for p in indices_map:
      d = count_map[p] / n
      
      if d > 0:
        h += d * math.log(d)

    h *= -1
    
    score_map[l] = h
    
  ret = np.zeros(len(labels))
  
  s = 0
  
  for l in label_ids:
    indices = label_indices_map[l]
    
    ret[indices] = score_map[l]
    
    s += len(indices)
    
  # LZ/DZ diversity
  ret_lz, score_map_lz, lz_map = shannon_diversity_mode(labels, indices_map)

  #
  # Write
  #
  
  f = open('diversity_{}.txt'.format(name), 'w')
  
  h1 = '{} patients'.format(';'.join([p for p in sorted(indices_map)]))
  
  f.write('cluster\tsize\t{}\tpatient diversity\tlz;dz sizes\tlz;dz diversity\n'.format(h1))
  
  for l in sorted(score_map):
    patients = ";".join([str(patient_map[l][p]) for p in sorted(indices_map)])
    
    modes = ";".join([str(lz_map[l]['lz']), str(lz_map[l]['dz'])])
    
    f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(l + 1, len(label_indices_map[l]), patients, score_map[l], modes, score_map_lz[l]))
  
  f.close()
  
  return ret, ret_lz




def correlation_plot(x, y, clusters, name, marker='o', s=MARKER_SIZE, xlabel='', ylabel='', fig=None, ax=None):
    """
    Create a tsne plot without the formatting
    """
    
    c = get_colors()
  
    if ax is None:
        fig, ax = libplot.new_fig()
      
    ids = list(sorted(set(clusters['Cluster'])))
      
    for i in range(0, len(ids)):
        l = ids[i]
        
        print('Label {}'.format(l))
        indices = np.where(clusters['Cluster'] == l)[0]
        
        n = len(indices)
        
        label = 'C{} ({:,})'.format(l, n)
    
        x1 = np.take(x, indices)
        y1 = np.take(y, indices)
          
        ax.scatter(x1, y1, color=c[i], edgecolor=c[i], s=s, marker=marker, alpha=0.8, label=label)
    
    sns.regplot(np.array(x), np.array(y), ax=ax, scatter=False)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    format_legend(ax)
    
    return fig, ax


def scatter_clusters(x,
                     y, 
                     clusters,
                     markers='o',
                     s=libplot.MARKER_SIZE,
                     alpha=libplot.ALPHA,
                     colors='none',
                     edgecolors='none',
                     linewidth=1,
                     prefix = '',
                     mode='plot',
                     fig=None,
                     ax=None,
                     sort=True,
                     cluster_order=None,
                     default_color = '#cccccc'):
    """
    Create a plot of clusters.
    
    Parameters
    ----------
    x : array
        x coordinates
    y : array
        y coordinates
    mode : str, optional
        Specify how to render plot. 
        'plot' - conventional graphics plot
        'text' - use markers to render text at locations using cluster color
        and marker to set the text color and text respectively. Thus a
        blue cluster with marker '1' will have all its points rendered as
        blue '1's rather than points.
    """
    
    if ax is None:
        fig, ax = libplot.new_fig()
    
    if colors is None:
        colors = get_colors()
    
    if cluster_order is None:    
        if sort:
            cluster_order = list(sorted(set(clusters['Cluster'])))
        else:
            cluster_order = []
            used = set()
            
            for id in clusters['Cluster']:
                if id not in used:
                    cluster_order.append(id)
                    used.add(id)
            
      
    for i, c in enumerate(cluster_order):
        indices = np.where(clusters['Cluster'] == c)[0]
        
        n = len(indices)
        
        label = '{}{} ({:,})'.format(prefix, c, n)
    
        x1 = x[indices] #np.take(x, indices)
        y1 = y[indices] #np.take(y, indices)
        
        if isinstance(colors, dict):
            color = colors.get(c, default_color)
        elif isinstance(colors, list):
            if c == 101:
                # special case where 101 is colored separately
                color = CLUSTER_101_COLOR
            else:
                color = colors[i]
        else:
            # fixed color
            color = colors
            
        print('scatter', c, color)
        
        if isinstance(markers, dict) and c in markers:
            marker = markers[c]
        else:
            marker = markers
            
        if isinstance(edgecolors, dict) and c in edgecolors:
            edgecolor = edgecolors[c]
        else:
            edgecolor = edgecolors
        
        if mode == 'text':
            ax.scatter(x1, y1, color='white', s=s, marker=marker, alpha=alpha, label=label)
            
            for li in range(0, x1.size):
                xl = x1[li]
                yl = y1[li]
                
                if marker == 's':
                    ax.text(xl, yl, '1', color=edgecolor)
                elif marker == '^':
                    ax.text(xl, yl, '2', color=edgecolor)
                elif marker == 'v':
                    ax.text(xl, yl, '3', color=edgecolor)
        else:
            ax.scatter(x1, 
                       y1, 
                       color=color, 
                       edgecolors=edgecolor,
                       linewidths=linewidth,
                       s=s, 
                       marker=marker, 
                       alpha=alpha, 
                       label=label)

    return fig, ax


def cluster_colors(clusters, colors=None):
    
    if colors is None:
        colors = get_colors()
        
    ret = []
    
    ids = list(sorted(set(clusters['Cluster'])))
    
    cmap = {}
    
    for i in range(0, len(ids)):
        cmap[ids[i]] = i
    
    for i in range(0, clusters.shape[0]):
        ret.append(colors[cmap[clusters['Cluster'][i]]])
        
    return ret





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
            sc[np.where(clusters.index.str.contains(id))[0]] = s
            c += 1

    df = pd.DataFrame(sc, index=clusters.index, columns=['Cluster'])

    return df




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

    scatter_clusters(d.iloc[:, dim1].values,
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
        format_legend(ax,
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
            format_legend(ax,
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
    #format_simple_axes(ax, title="t-SNE")
    #format_legend(ax, cols=6, markerscale=2)

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
                     outline=False,
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

    if outline:
        cluster_outline(tsne,
                cluster,
                clusters,
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
        colors = get_colors()

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
                 order=None):
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

    if order is None:
        order = list(sorted(set(clusters['Cluster'])))

    order = np.array(order)

    n = order.size

    if cols == -1:
        cols = int(np.ceil(np.sqrt(n)))

    rows = int(np.ceil(n / cols))

    w = size * cols
    h = size * rows

    fig = libplot.new_base_fig(w=w, h=h)

    if colors is None:
        colors = get_colors()

    # Where to plot figure
    pc = 1

    for i, cluster in enumerate(order):
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
                        order=None,
                        method='tsne',
                        dir='.',
                        out=None):
    fig = cluster_grid(tsne,
                       clusters,
                       colors=colors,
                       cols=cols,
                       size=size,
                       add_titles=add_titles,
                       order=order)

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
#        colors = colors()
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