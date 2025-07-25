# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:04:55 2023

@author: kim uittenhove
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler
from utils import save_pickle_data, load_pickle_data

def scale_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def cleanup_noisy_points(df):
    """
    Pass a window over each sequence of 10 sequential points.
    Remove cluster IDs that are not part of the majority cluster.

    Parameters:
    df (pd.DataFrame): DataFrame containing cluster data.

    Returns:
    pd.DataFrame: DataFrame with cleaned cluster data.
    """
    clustered_indices = df[df['resample'] == 1].index
    for start in range(0, len(clustered_indices), 10):
        section_indices = clustered_indices[start:start + 10]
        section = df.loc[section_indices]
        majority_cluster = section['cluster'].mode().iloc[0]
        df.loc[section[section['cluster'] != majority_cluster].index, 'cluster'] = np.nan
    return df

def cluster_directions(df, min_cluster_size):
    """
    Cluster the directions and positions using HDBSCAN.

    Parameters:
    df (pd.DataFrame): DataFrame containing direction and position data.
    min_cluster_size (int): Minimum size of clusters.

    Returns:
    pd.DataFrame: DataFrame with cluster labels.
    """
    directions = df[['dir_h', 'dir_v', 'X_smooth', 'Y_smooth']][df['resample'] == 1]
    scaled_directions = scale_data(directions)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(scaled_directions)
    
    df['cluster'] = np.nan
    df.loc[df['resample'] == 1, 'cluster'] = cluster_labels
    return cleanup_noisy_points(df)

def identify_points_with_cluster_sequences(df, cluster_sequences):
    """
    Fill in the linepoints variable to contain continuous sections of cluster attributions
    Determine turnpoints as points betweene line sections

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    cluster_sequences (pd.DataFrame): DataFrame containing cluster sequences.

    Returns:
    pd.DataFrame: DataFrame with linepoints and turnpoints marked.
    """
    df['linepoint'] = np.nan
    df['turnpoint'] = 0

    def find_nearest_lift_0_point(index):
        nearest = min(df.index, key=lambda x: abs(x - index) if df.loc[x, 'lift'] == 0 else float('inf'))
        return nearest

    def add_turn_point(index):
        if 0 <= index < len(df):
            df.loc[index, 'turnpoint'] = 1

    prev_end_idx = None

    for _, row in cluster_sequences.iterrows():
        cluster_id, start_idx, end_idx = row['cluster_id'], row['start_idx'], row['end_idx']
        df.loc[start_idx:end_idx, 'linepoint'] = cluster_id

        if prev_end_idx is not None:
            mid_idx = (prev_end_idx + start_idx) // 2
            nearest_mid_idx = find_nearest_lift_0_point(mid_idx)
            add_turn_point(nearest_mid_idx)

        prev_end_idx = end_idx

    if not cluster_sequences.empty:
        add_turn_point(cluster_sequences['start_idx'].min() - 1)
        add_turn_point(cluster_sequences['end_idx'].max() + 1)

    return df

def extract_cluster_sequences(df):
    """
    Extract sequences of cluster IDs along with their start and end indices.

    Parameters:
    df (pd.DataFrame): DataFrame containing cluster data.

    Returns:
    pd.DataFrame: DataFrame with cluster sequences.
    """
    cluster_sequences = []
    clusters = df['cluster'].dropna().sort_index()
    prev_cluster = None
    start_idx = None

    for idx, cluster in clusters.items():
        if cluster != prev_cluster:
            if prev_cluster is not None:
                cluster_sequences.append({'cluster_id': prev_cluster, 'start_idx': start_idx, 'end_idx': idx - 1})
            start_idx = idx
            prev_cluster = cluster

    if prev_cluster is not None:
        cluster_sequences.append({'cluster_id': prev_cluster, 'start_idx': start_idx, 'end_idx': clusters.index[-1]})
        
    return pd.DataFrame(cluster_sequences).query('cluster_id != -1')

def plot_clusters(df):
    plt.figure(figsize=(8, 6))
    clustered = df[df['cluster'] != -1]
    noise = df[df['cluster'] == -1]
    plt.scatter(clustered['X_smooth'], clustered['Y_smooth'], s=30, c=clustered['cluster'], cmap='tab20', label='Clustered Points')
    plt.scatter(noise['X_smooth'], noise['Y_smooth'], s=50, marker='x', c='black', label='Cluster -1')
    plt.legend()
    plt.title(df['id'].iloc[0])
    plt.xlabel('X-Smooth')
    plt.ylabel('Y-Smooth')
    plt.grid(True)
    plt.show()
   
def plot_points(df):
    plt.figure(figsize=(8, 8))
    plt.plot(df['X_smooth'], df['Y_smooth'], color='grey')
    colors = ['red', 'blue']
    prev_turnpoint_idx = None
    color_idx = 0

    for turnpoint_idx in df[df['turnpoint'] == 1].index:
        if prev_turnpoint_idx is not None:
            plt.plot(df['X_smooth'].iloc[prev_turnpoint_idx:turnpoint_idx + 1],
                     df['Y_smooth'].iloc[prev_turnpoint_idx:turnpoint_idx + 1],
                     color=colors[color_idx % len(colors)])
            color_idx += 1
        prev_turnpoint_idx = turnpoint_idx

    plt.scatter(df[df['turnpoint'] == 1]['X_smooth'], df[df['turnpoint'] == 1]['Y_smooth'], color='green', label='Turnpoints')
    plt.legend()
    plt.title(df['id'][0])
    plt.show()
    
def process_data_turning_pts(input_data, keys_to_process):
    processed_data = {}
    for key in keys_to_process:
        if key in input_data:
            processed_dataset = []
            for df in input_data[key]:
                df = cluster_directions(df, min_cluster_size=20)
                cluster_sequences = extract_cluster_sequences(df)
                df = identify_points_with_cluster_sequences(df, cluster_sequences)
                plot_points(df)
                processed_dataset.append(df)

            processed_data[key] = processed_dataset
    return processed_data

# Define the output path and load the processed data
OUTPUT_PATH = ''
loaded_data = load_pickle_data(os.path.join(OUTPUT_PATH, 'preprocessed_data.pkl'))

# Specify the keys to process
keys_to_process = ['tmt_a', 'tmt_b','tmt_a_long', 'tmt_b_long']
data_turning_pts = process_data_turning_pts(loaded_data, keys_to_process)

# Save the processed data to a pickle file
turning_pts_pickle_file_name = 'processed_data_t_pts_dir.pkl'
save_pickle_data(data_turning_pts, os.path.join(OUTPUT_PATH, turning_pts_pickle_file_name))
