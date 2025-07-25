# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:16:59 2024

Author: kim uittenhove
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import cdist
from utils import save_pickle_data, load_pickle_data, max_thresholds

N = 5
threshold_values = {
    condition: list(zip(np.linspace(0, h, N), np.linspace(0, v, N)))
    for condition, (h, v) in max_thresholds.items()
}

def scale_data(real_data, target_data):
    """
    Scale the real and target data using MinMaxScaler.

    Parameters:
    real_data (np.ndarray): Array containing the data collected by participants.
    target_data (np.ndarray): Array containing the theoretical target data.

    Returns:
    tuple: Scaled real and target data.
    """
    scaler = MinMaxScaler()
    real_data_scaled = scaler.fit_transform(real_data)
    target_data_scaled = scaler.fit_transform(target_data)
    return real_data_scaled, target_data_scaled

def find_best_reflection(points, target_points):
    """
    Find the best reflection (flipping) for points 
    Minimize the distance between turning points in participant data and real target points

    Parameters:
    points (np.ndarray): Array of points to be reflected.
    target_points (np.ndarray): Array of target points.

    Returns:
    tuple: Best flip configuration for x and y coordinates.
    """
    def flip_points(points, flip_x, flip_y):
        flipped_points = points.copy()
        if flip_x:
            flipped_points[:, 0] = 1 - flipped_points[:, 0]
        if flip_y:
            flipped_points[:, 1] = 1 - flipped_points[:, 1]
        return flipped_points

    def compute_cost(flipped_points, target_points):
        cost = 0
        for flipped_point in flipped_points:
            min_dist = np.min(cdist([flipped_point], target_points, 'sqeuclidean'))
            cost += min_dist
        return cost

    min_cost = float('inf')
    best_flip_x, best_flip_y = False, False

    flip_combinations = list(itertools.product([False, True], repeat=2))
    for flip_x, flip_y in flip_combinations:
        flipped_points = flip_points(points, flip_x, flip_y)
        cost = compute_cost(flipped_points, target_points)
        if cost < min_cost:
            min_cost = cost
            best_flip_x = flip_x
            best_flip_y = flip_y

    return best_flip_x, best_flip_y

def apply_reflection(points, flip_x, flip_y):
    """
    Apply reflection (flipping) to points based on the given configuration.

    Parameters:
    points (np.ndarray): Array of points to be reflected.
    flip_x (bool): Whether to flip x coordinates.
    flip_y (bool): Whether to flip y coordinates.

    Returns:
    np.ndarray: Reflected points.
    """
    def flip_points(points, flip_x, flip_y):
        flipped_points = points.copy()
        if flip_x:
            flipped_points[:, 0] = 1 - flipped_points[:, 0]
        if flip_y:
            flipped_points[:, 1] = 1 - flipped_points[:, 1]
        return flipped_points

    return flip_points(points, flip_x, flip_y)

def is_point_in_ellipse(point, center, horizontal_threshold, vertical_threshold):
    """
    Check if a point is within an ellipse defined by the center and thresholds.

    Parameters:
    point (tuple): Coordinates of the point.
    center (tuple): Coordinates of the ellipse center.
    horizontal_threshold (float): Horizontal threshold of the ellipse.
    vertical_threshold (float): Vertical threshold of the ellipse.

    Returns:
    bool: True if the point is within the ellipse, False otherwise.
    """
    epsilon = 1e-10  # A small value to prevent division by zero
    x, y = point
    h, k = center
    return ((x - h) ** 2) / (max(horizontal_threshold ** 2, epsilon)) + ((y - k) ** 2) / (max(vertical_threshold ** 2, epsilon)) <= 1

def find_turning_points_corresponding_to_targets(df_scaled, df, target_data, target_center_keys, condition, subject_id):
    """
    Find turning points corresponding to theoretical targets
    Use dynamically increasing thresholds
    Also select the closest non-turning point, to be used when no turning points match

    Parameters:
    df_scaled (np.ndarray): Scaled DataFrame of coordinates.
    df (pd.DataFrame): Original DataFrame.
    target_data (np.ndarray): Array of target coordinates.
    target_center_keys (list): List of target center keys.
    condition (str): Experimental condition.
    subject_id (str): Subject identifier.

    Returns:
    pd.DataFrame: DataFrame with potential targets and corresponding thresholds.
    """
    t_pts_coords_direction = df_scaled[df.turnpoint == 1]
    all_data_coords = df_scaled

    thresholds = threshold_values[condition]
    max_threshold_horizontal, max_threshold_vertical = max_thresholds[condition]

    # DataFrame to store potential targets and thresholds
    potential_targets_df = pd.DataFrame(index=target_center_keys, columns=['Potential Targets', 'Threshold'])

    # Initialize lists for each target
    for key in target_center_keys:
        potential_targets_df.at[key, 'Potential Targets'] = []
        potential_targets_df.at[key, 'Threshold'] = []

    for i, target_point in enumerate(target_data):
        target_key = target_center_keys[i]
        identified_indices = set()  # Keep track of identified turning points

        # Loop over each threshold
        for threshold in thresholds:
            horizontal_threshold, vertical_threshold = threshold

            # Check for turning points within the threshold 
            for tp_idx, tp_coords in enumerate(t_pts_coords_direction):
                if is_point_in_ellipse(tp_coords, target_point, horizontal_threshold, vertical_threshold):
                    index = np.where(df.turnpoint == 1)[0][tp_idx]
                    if index not in identified_indices:  # Only add if not already identified
                        identified_indices.add(index)
                        potential_targets_df.at[target_key, 'Potential Targets'].append(index)
                        potential_targets_df.at[target_key, 'Threshold'].append(threshold)

        # Add the closest point within the max threshold as the last potential target
        # Find the closest point within the max threshold
        closest_point_idx = None
        min_dist = float('inf')
        for idx, point in enumerate(all_data_coords):
            if is_point_in_ellipse(point, target_point, max_threshold_horizontal, max_threshold_vertical):
                if df.loc[df.index[idx], 'lift'] == 0 and idx not in identified_indices:
                    dist = np.linalg.norm(point - target_point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point_idx = idx

        if closest_point_idx is not None:
            potential_targets_df.at[target_key, 'Potential Targets'].append(closest_point_idx)
            potential_targets_df.at[target_key, 'Threshold'].append((max_threshold_horizontal, max_threshold_vertical))
    
    return potential_targets_df

def plot_trajectory(df_scaled, target_centers_scaled, potential_targets_df, subject_id, condition, turn_points):
    plt.figure(figsize=(10, 8))
    
    # Plot trajectory
    plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c='lightgray', label='Data Points', s=2)
    
    # Plot turn points
    plt.scatter(turn_points[:, 0], turn_points[:, 1], c='green', marker='o', s=100, label='Turn Points')
    
    # Plot theoretical targets
    plt.scatter(target_centers_scaled[:, 0], target_centers_scaled[:, 1], c='blue', s=100, label='Theoretical Targets')
    for i, (x, y) in enumerate(target_centers_scaled):
        plt.annotate(f'T{i+1}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14, color='blue')
    
    # Plot potential targets
    for target_key, row in potential_targets_df.iterrows():
        potential_indices = row['Potential Targets']
        if potential_indices:
            plt.scatter(df_scaled[potential_indices, 0], df_scaled[potential_indices, 1], marker='x', color='red', s=10)
            for idx in potential_indices:
                plt.annotate(f'{target_key}', (df_scaled[idx, 0], df_scaled[idx, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14, color='red')

    plt.title(f'Subject {subject_id} - Condition: {condition}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def process_data(t_pts_direction, keys_to_process):
    potential_targets_data = {}
    scaled_data = {}
    scaled_target_centers = {}
    
    for key in keys_to_process:
        if key in t_pts_direction:
            potential_targets_dataset = {}
            scaled_dataset = []
            
            target_centers_for_key = np.array([coord for _, coord in target_centers[key]], dtype=np.float64)

            for df in t_pts_direction[key]:
                df_scaled, target_centers_scaled = scale_data(df[['X_smooth', 'Y_smooth']].values, target_centers_for_key)
                turning_points_scaled = df_scaled[df['turnpoint'].values == 1]
                
                # Correct reflections
                best_flip_x, best_flip_y = find_best_reflection(turning_points_scaled, target_centers_scaled)
                turning_points_scaled = apply_reflection(turning_points_scaled, best_flip_x, best_flip_y)
                df_scaled = apply_reflection(df_scaled, best_flip_x, best_flip_y)
                
                subject_id = df['id'].iloc[0]
                target_center_keys = np.array([key for key, _ in target_centers[key]])
                potential_targets_df = find_turning_points_corresponding_to_targets(df_scaled, df, target_centers_scaled, target_center_keys, key, subject_id)
        
                # Plot the trajectory with theoretical and potential targets
                plot_trajectory(df_scaled, target_centers_scaled, potential_targets_df, subject_id, key, turning_points_scaled)

                potential_targets_dataset[subject_id] = potential_targets_df
                scaled_dataset.append(df_scaled)
            
            potential_targets_data[key] = potential_targets_dataset
            scaled_data[key] = scaled_dataset
            scaled_target_centers[key] = target_centers_scaled  # Correctly store the scaled target centers

    return potential_targets_data, scaled_data, scaled_target_centers

# Replace OUTPUT_PATH with the actual path to your pickled files
OUTPUT_PATH = ''

# Load target centers from target_centers.npz file
target_centers_file = os.path.join(OUTPUT_PATH, 'target_centers.npz')
target_centers_data = np.load(target_centers_file, allow_pickle=True)
target_centers = {k: target_centers_data[k] for k in target_centers_data.files}

t_pts_direction = load_pickle_data(os.path.join(OUTPUT_PATH, 'processed_data_t_pts_dir.pkl'))

# Remove potential key suffixes to access target centers
suffixes_to_remove = [""]
t_pts_direction = {key.rstrip("".join(suffixes_to_remove)): value for key, value in t_pts_direction.items()}

keys_to_process = ['tmt_a', 'tmt_b', 'tmt_a_long', 'tmt_b_long']
potential_targets_data, scaled_data, scaled_target_centers = process_data(t_pts_direction, keys_to_process)

# Save the processed data as a single pickled file
save_pickle_data(potential_targets_data, os.path.join(OUTPUT_PATH, 'data_potential_target_pts.pkl'))
save_pickle_data(scaled_data, os.path.join(OUTPUT_PATH, 'scaled_position_data.pkl'))
save_pickle_data(scaled_target_centers, os.path.join(OUTPUT_PATH, 'scaled_target_centers.pkl'))
