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


def create_merge_cluster_info(clusters, name, sample_names=('RK10001', 'RK10002'), dir='.'):
    """
    Summarizes how many samples are in each cluster and from which experiment
    they came.
    
    Parameters
    ----------
    clusters : DataFrame
        table of clusters.
    name : str
        prefix for file output
    """
    
    cids = list(sorted(set(clusters['Cluster'].tolist())))

    samples = np.array(['' for i in range(0, clusters.shape[0])], dtype=object)
  
    for i in range(0, len(sample_names)):
        id = '-{}'.format(i + 1)
        
        samples[clusters.index.str.contains(id)] = sample_names[i]
        id = '.{}'.format(i + 1)
        samples[clusters.index.str.contains(id)] = sample_names[i]
 
    size_map = {}
    cluster_sample_sizes = collections.defaultdict(lambda : collections.defaultdict(int))
    
    for cid in cids:
        size_map[cid] = clusters[clusters['Cluster'] == cid]['Cluster'].shape[0]
        
        for i in range(0, len(sample_names)):
            id = '-{}'.format(i + 1)
            # count how many cells are in each cluster for each sample
            cluster_sample_sizes[cid][sample_names[i]] = clusters[(clusters['Cluster'] == cid) & clusters.index.str.contains(id)].shape[0]
            id = '.{}'.format(i + 1)
            cluster_sample_sizes[cid][sample_names[i]] = clusters[(clusters['Cluster'] == cid) & clusters.index.str.contains(id)].shape[0]
            

    sample_counts = np.zeros((clusters.shape[0], len(sample_names)), dtype=int)
    sizes = np.zeros(clusters.shape[0], dtype=int)

    for i in range(0, clusters.shape[0]):
        cs = cluster_sample_sizes[clusters['Cluster'][i]]
        
        for j in range(0, len(sample_names)):
            sample_counts[i, j] = cs[sample_names[j]]
        
        
        sizes[i] = size_map[clusters['Cluster'][i]]
        
       
    df = pd.DataFrame({'Barcode':clusters.index, 'Cluster':clusters['Cluster'], 'Sample':samples, 'Size':sizes})
    
    for i in range(0, len(sample_names)):
        df['Count {}'.format(sample_names[i])] = sample_counts[:, i]
    
    #df = df[['Barcode', 'Cluster', 'Sample', '{} count'.format(sample_names[0]), '{} count'.format(sample_names[1]), 'Size']]
    df.to_csv('{}/{}_cell_cluster_info.txt'.format(dir, name), sep='\t', header=True, index=False)
    
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
    df.to_csv('{}/{}_cluster_info.txt'.format(dir, name), sep='\t', header=True, index=False)
    


def create_origin_cluster_info(clusters, name, dir='.'):
    """
    Summarizes how many samples are in each cluster and from which experiment
    they came.
    
    Parameters
    ----------
    clusters : DataFrame
        table of clusters.
    name : str
        prefix for file output
    """
    
    cids = list(sorted(set(clusters['Cluster'].tolist())))
    
    sample_names = list(sorted(set(clusters['origin'].tolist())))

    samples = clusters['origin'] #np.array(['' for i in range(0, clusters.shape[0])], dtype=object)
 
#    for i in range(0, len(sample_names)):
#        id = '-{}'.format(i + 1)
#        samples[clusters.index.str.contains(id)] = sample_names[i]
# 
    size_map = {}
    cluster_sample_sizes = collections.defaultdict(lambda : collections.defaultdict(int))
    
    for cid in cids:
        size_map[cid] = clusters[clusters['Cluster'] == cid]['Cluster'].shape[0]
        
        for sample_name in sample_names:
            
            if cid == 6 and sample_name == '1_4_gc_b_c11':
                print(sample_name)
                hmm = clusters[(clusters['Cluster'] == cid) & (clusters['origin'] == sample_name)]
                print(hmm.index.values.size)
                
            # count how many cells are in each cluster for each sample
            cluster_sample_sizes[cid][sample_name] = clusters[(clusters['Cluster'] == cid) & (clusters['origin'] == sample_name)].shape[0]
    
    print('---')
    sample_counts = np.zeros((clusters.shape[0], len(sample_names)), dtype=int)
    sizes = np.zeros(clusters.shape[0], dtype=int)

    for i in range(0, clusters.shape[0]):
        cs = cluster_sample_sizes[clusters['Cluster'][i]]
        
        for j in range(0, len(sample_names)):
            sample_counts[i, j] = cs[sample_names[j]]
        
        
        sizes[i] = size_map[clusters['Cluster'][i]]
        
       
    df = pd.DataFrame({'Barcode':clusters.index, 'Cluster':clusters['Cluster'], 'Sample':samples, 'Size':sizes})
    
    for i in range(0, len(sample_names)):
        df['Count {}'.format(sample_names[i])] = sample_counts[:, i]
    
    #df = df[['Barcode', 'Cluster', 'Sample', '{} count'.format(sample_names[0]), '{} count'.format(sample_names[1]), 'Size']]
    df.to_csv('{}/{}_cell_origin_info.txt'.format(dir, name), sep='\t', header=True, index=False)
    
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
    df.to_csv('{}/{}_origin_info.txt'.format(dir, name), sep='\t', header=True, index=False)
