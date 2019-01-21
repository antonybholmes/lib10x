#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:43:43 2018

@author: antony
"""

import pandas as pd
import numpy as np
import collections


def get_tsne_plot_name(name, t1=1, t2=2):
    return 'tsne_sample_{}.pdf'.format(name)


def create_sample_clusters(tsne, labels):
    """
    Create a cluster data frame from samples.
    
    Parameters
    ----------
    tsne : pandas.DataFrame
        t-sne table
    labels : list or tuple
        String labels for sample names
        
    Returns
    -------
    pandas.DataFrame
        Cells with cluster label for which sample they belong to.
    """
    id = -1
    
    clusters = np.array(['' for i in range(0, tsne.shape[0])], dtype=object)
    
    for label in labels:
        clusters[tsne.index.str.contains('{}'.format(id))] = label
        id -= 1
        
    df = pd.DataFrame({'Barcode':tsne.index, 'Cluster':clusters})
    df = df.set_index('Barcode')
    
    return df

def create_merge_cluster_info(counts, clusters, name, sample_names=('RK10001', 'RK10002')):
    """
    Summarizes how many samples are in each cluster and from which experiment
    they came.
    
    Parameters
    ----------
    counts : DataFrame
        table of counts.
    clusters : DataFrame
        table of clusters.
    name : str
        prefix for file output
    """
    
    cids = list(sorted(set(clusters['Cluster'].tolist())))

    samples = np.array(['' for i in range(0, counts.columns.shape[0])], dtype=object)
 
    for i in range(0, len(sample_names)):
        id = '-{}'.format(i + 1)
        samples[counts.columns.str.contains(id)] = sample_names[i]
 
    size_map = {}
    cluster_sample_sizes = collections.defaultdict(lambda : collections.defaultdict(int))
    
    for cid in cids:
        size_map[cid] = clusters[clusters['Cluster'] == cid]['Cluster'].shape[0]
        
        for i in range(0, len(sample_names)):
            id = '-{}'.format(i + 1)
            
            # count how many cells are in each cluster for reach sample
            cluster_sample_sizes[cid][sample_names[i]] = clusters[(clusters['Cluster'] == cid) & counts.columns.str.contains(id)].shape[0]
            

    sample_counts = np.zeros((counts.columns.shape[0], len(sample_names)), dtype=int)
    sizes = np.zeros(counts.columns.shape[0], dtype=int)

    for i in range(0, counts.columns.shape[0]):
        cs = cluster_sample_sizes[clusters['Cluster'][i]]
        
        for j in range(0, len(sample_names)):
            sample_counts[i, j] = cs[sample_names[j]]
        
        
        sizes[i] = size_map[clusters['Cluster'][i]]
        
       
    df = pd.DataFrame({'Barcode':counts.columns, 'Cluster':clusters['Cluster'], 'Sample':samples, 'Size':sizes})
    
    for i in range(0, len(sample_names)):
        df['Count {}'.format(sample_names[i])] = sample_counts[:, i]
    
    #df = df[['Barcode', 'Cluster', 'Sample', '{} count'.format(sample_names[0]), '{} count'.format(sample_names[1]), 'Size']]
    df.to_csv('{}_cell_cluster_info.txt'.format(name), sep='\t', header=True, index=False)
    
    # table of cluster sizes
    
    sample_counts = np.zeros((len(cids), len(sample_names)), dtype=int)
    sizes = np.zeros(len(cids), dtype=int)

    for i in range(0, len(cids)):
        c = cids[i]
        
        cs = cluster_sample_sizes[c]
        
        for j in range(0, len(sample_names)):
            sample_counts[i, j] = cs[sample_names[j]]
        
        sizes[i] = size_map[c]
    
    df = pd.DataFrame({'Cluster':cids, 'Size':sizes})
    
    for i in range(0, len(sample_names)):
        df['Count {}'.format(sample_names[i])] = sample_counts[:, i]
    
    #df = df[['Barcode', 'Cluster', 'Sample', '{} count'.format(sample_names[0]), '{} count'.format(sample_names[1]), 'Size']]
    df.to_csv('{}_cluster_info.txt'.format(name), sep='\t', header=True, index=False)
    

