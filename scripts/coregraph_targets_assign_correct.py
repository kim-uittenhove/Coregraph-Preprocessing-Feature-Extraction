
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:52:55 2024

This script processes trajectory data to correct target connections based on given rules and exceptions.
It then plots the corrected trajectories and saves the processed data.

@author: kim uittenhove
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from utils import save_pickle_data, load_pickle_data, max_thresholds

# Global definition of exceptions for specific conditions
EXCEPTIONS = {
    'tmt_a_long': [(22, 24, 23), (24, 3, 25), (20, 19, 21), (12, 8, 13), (21, 19, 22), (3, 1, 4), (23, 25, 24), (10, 2, 11), (11, 9, 12), (18, 20, 19)],
    'tmt_b_long': [(3, 5, 4), (14, 16, 15), (17, 4, 18), (21, 12, 22), (22, 24, 23), (24, 14, 25), (1, 6, 2), (5, 1, 6), (9, 20, 10), (12, 3, 13),
                   (12, 14, 13), (13, 3, 14), (13, 16, 14), (14, 16, 15), (15, 17, 18), (19, 8, 20), (7, 18, 8), (12, 24, 13), (21, 10, 22), (3, 13, 4),
                   (23, 14, 24), (24, 16, 25), (20, 10, 21), (17, 7, 18), (12, 16, 13), (9, 2, 10), (15, 23, 14)]
    # Add more conditions and their exception triples as needed
}

def get_missing_target_ids(df, potential_targets_df):
    """
    Identify missing target IDs by checking which targets have not been assigned.
    
    Parameters:
        df (DataFrame): DataFrame containing trajectory data.
        potential_targets_df (DataFrame): DataFrame containing potential targets.
    
    Returns:
        list: List of missing target IDs.
    """
    assigned_target_ids = df['target_id'].unique()
    missing_target_ids = [
        target_id for target_id in potential_targets_df.index
        if target_id not in assigned_target_ids and 'unused' in potential_targets_df.at[target_id, 'Status']
    ]
    return missing_target_ids

def assign_and_correct_targets(df, potential_targets_df, condition):
    """
    Assign target IDs to trajectory data based on potential targets and correct the assignments iteratively.

    This function iterates over missing theoretical target IDs and assigns them to the df, using the single most suitable
    target provided in the potential targets DataFrame.
    After each assignment iteration, it calls the `correct_target_assignments` function to
    correct the assignments based on the expected sequence, taking into account some exceptions. 
    Following this step, it is possible that some theoretical target IDs are unassigned and need to be assigned again.

    Therefore, the process continues until all missing theoretical target IDs are assigned or until all potential targets have been
    exhausted.

    Parameters:
        trajectory_df (DataFrame): DataFrame containing trajectory data.
        potential_targets_df (DataFrame): DataFrame containing potential targets.
        condition (str): Condition key to check for exceptions during the correction process.

    Returns:
        DataFrame: DataFrame with assigned and corrected target IDs.
    """
    df['connect_correct'] = -1  # Initialize connection correctness column
    df['target_id'] = -1  # Initialize target ID column

    # Initialize the status of each potential target in potential_targets_df
    potential_targets_df['Status'] = potential_targets_df.apply(
        lambda x: ['unused' for _ in x['Potential Targets']], axis=1)

    missing_target_ids = get_missing_target_ids(df, potential_targets_df)
    
    # Loop until there are no missing target_ids or until all potential targets have been tried
    while missing_target_ids:
        for target_id in missing_target_ids:
            potential_indices = potential_targets_df.at[target_id, 'Potential Targets']
            status_list = potential_targets_df.at[target_id, 'Status']

            for idx, (potential_index, status) in enumerate(zip(potential_indices, status_list)):
                if status == 'unused' and df.at[potential_index, 'target_id'] == -1:
                    # Find if there's an existing assignment and mark it as 'unassigned'
                    for j, s in enumerate(status_list):
                        if s == 'assigned':
                            potential_targets_df.at[target_id, 'Status'][j] = 'unassigned'
                            break

                    # Assign the new target
                    df.at[potential_index, 'target_id'] = target_id
                    df.at[potential_index, 'linepoint'] = np.nan
                    potential_targets_df.at[target_id, 'Status'][idx] = 'assigned'
                    break  # Assign and break
                elif status == 'unused':
                    potential_targets_df.at[target_id, 'Status'][idx] = 'skipped'

        # Run the correction logic after assignment
        correct_target_assignments(df, condition)

        # Update the list of missing target_ids for the next iteration
        missing_target_ids = get_missing_target_ids(df, potential_targets_df)

    return df

def correct_target_assignments(df, condition):
    """
    Correct the target assignments in the trajectory data based on the expected correct sequence.
    Also takes into account some exceptions.

    This function iterates over each row in df and checks the current target ID. It
    determines if the current target ID follows the expected sequence based on the last non-exception target ID.

    If the current target ID does not follow the expected sequence, it checks if the sequence meets an exception
    for the given condition using the `meets_exception` function. If an exception is met, the target ID is reset
    to -1, and the `connect_correct` column is set to -1. If an exception is not met, the `connect_correct`
    column is set to 0, indicating an incorrect connection.

    Parameters:
        trajectory_df (DataFrame): DataFrame containing trajectory data.
        condition (str): Condition key to check for exceptions during the correction process.
    """
    last_non_exception_target_id = None  # To track the last non-exception target ID

    # Extract valid target IDs and their indices
    valid_targets = [(index, row['target_id']) for index, row in df.iterrows() if row['target_id'] != -1]

    for i, row in df.iterrows():
        current_target_id = row['target_id']
            
        if current_target_id == -1:
            continue
            
        # If we have not seen any valid target ID before
        if last_non_exception_target_id is None:
            df.at[i, 'connect_correct'] = 1
            last_non_exception_target_id = current_target_id
            continue

        # Get the previous and next valid target IDs
        previous_target_id, next_target_id = find_nearest_valid_targets(i, valid_targets)
            
        # If the current target ID follows the last non-exception target ID or is the same
        if current_target_id in [last_non_exception_target_id, last_non_exception_target_id + 1]:
            df.at[i, 'connect_correct'] = 1
            last_non_exception_target_id = current_target_id
        else:
            # Check if the sequence meets an exception for the condition
            if meets_exception(previous_target_id, current_target_id, next_target_id, condition):
                df.at[i, 'target_id'] = -1
                df.at[i, 'connect_correct'] = -1
            else:
                df.at[i, 'connect_correct'] = 0
                last_non_exception_target_id = current_target_id

def find_nearest_valid_targets(index, valid_targets):
    """
    Find the nearest valid target IDs before and after the given index.
    
    Parameters:
        index (int): Index to check for nearest valid targets.
        valid_targets (list): List of tuples with valid target indices and IDs.
    
    Returns:
        tuple: Previous and next valid target IDs.
    """
    previous_target_id = next((target_id for idx, target_id in reversed(valid_targets) if idx < index), None)
    next_target_id = next((target_id for idx, target_id in valid_targets if idx > index), None)
    return previous_target_id, next_target_id

def meets_exception(previous_target_id, current_target_id, next_target_id, condition):
    """
    Check if the given target sequence meets any defined exceptions for the condition.
    
    Parameters:
        previous_target_id (int): Previous target ID.
        current_target_id (int): Current target ID.
        next_target_id (int): Next target ID.
        condition (str): Condition key to check for exceptions.
    
    Returns:
        bool: True if the sequence meets an exception, False otherwise.
    """
    # Use the global EXCEPTIONS dictionary
    return (previous_target_id, current_target_id, next_target_id) in EXCEPTIONS.get(condition, [])

def plot_corrected_points(df, scaled_position_data, scaled_target_centers, key, subject_id):
    """
    Plot the corrected trajectory points.
    
    Parameters:
        df (DataFrame): DataFrame containing trajectory data.
        scaled_position_data (array): Scaled position data.
        scaled_target_centers (dict): Dictionary of scaled target centers.
        key (str): Condition key.
        subject_id (str): Subject ID.
    """
    
    plot_dir = os.path.join('corrections', str(subject_id))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"correction_{subject_id}_{key}.png")
    
    # Ensure scaled_position_data is a numpy array
    scaled_position_data = np.asarray(scaled_position_data)
    # Get correct and incorrect indices
    correct_indices = df[df['connect_correct'] == 1].index
    incorrect_indices = df[df['connect_correct'] == 0].index
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the scaled position data
    ax.plot(scaled_position_data[:, 0], scaled_position_data[:, 1], 'k-', alpha=0.3, label='Trajectory')
    
    # Plot theoretical targets and ellipses
    target_centers = np.array(scaled_target_centers[key])
    max_threshold_horizontal, max_threshold_vertical = max_thresholds[key]
    ax.scatter(target_centers[:, 0], target_centers[:, 1], c='blue', label='Theoretical Targets')
    for i, (x, y) in enumerate(target_centers):
        ax.annotate(f'T{i+1}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14, color='blue')
        ellipse = patches.Ellipse((x, y), width=max_threshold_horizontal * 2, height=max_threshold_vertical * 2, edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(ellipse)
    
    # Plot correct and incorrect points
    ax.scatter(scaled_position_data[correct_indices, 0], scaled_position_data[correct_indices, 1], c='green', label='Correct', s=100)
    ax.scatter(scaled_position_data[incorrect_indices, 0], scaled_position_data[incorrect_indices, 1], c='red', label='Incorrect', s=100)
    
    # Set labels and title
    ax.set_xlabel('X_smooth')
    ax.set_ylabel('Y_smooth')
    ax.set_title(f'Corrected Points Trajectory\nKey: {key}, Subject ID: {subject_id}')
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

def process_data(t_pts_direction, potential_targets_data, keys_to_process):
    """
    Process trajectory data for each key and plot corrected points.
    
    Parameters:
        t_pts_direction (dict): Dictionary of trajectory data.
        potential_targets_data (dict): Dictionary of potential targets.
        keys_to_process (list): List of condition keys to process.
    
    Returns:
        dict: Dictionary of processed data.
    """
    processed_data = {}
    for key in keys_to_process:
        if key in t_pts_direction:
            processed_dataset = []
            
            for df_index, df in enumerate(t_pts_direction[key]):
                subject_id = df['id'].iloc[0]
                potential_targets_df = potential_targets_data[key][subject_id]
                df = assign_and_correct_targets(df, potential_targets_df, key)
                plot_corrected_points(df, scaled_position_data[key][df_index], scaled_target_centers, key, subject_id)                
                
                processed_dataset.append(df)
                
            processed_data[key] = processed_dataset
                
    return processed_data

# Replace OUTPUT_PATH with the actual path to your pickled files
OUTPUT_PATH = ''

t_pts_direction = load_pickle_data(os.path.join(OUTPUT_PATH, 'processed_data_t_pts_dir_YA.pkl'))
potential_targets_data = load_pickle_data(os.path.join(OUTPUT_PATH, 'data_potential_target_pts_YA.pkl'))
scaled_position_data = load_pickle_data(os.path.join(OUTPUT_PATH, 'scaled_position_data_YA.pkl'))
scaled_target_centers = load_pickle_data(os.path.join(OUTPUT_PATH, 'scaled_target_centers.pkl'))

# Remove the key suffixes
suffixes_to_remove = ["_YA"]
t_pts_direction = {key.rstrip("".join(suffixes_to_remove)): value for key, value in t_pts_direction.items()}

keys_to_process = ['tmt_a', 'tmt_b', 'tmt_a_long', 'tmt_b_long']
corrected_targets_data = process_data(t_pts_direction, potential_targets_data, keys_to_process)

# Save the processed_data as a single pickled file
save_pickle_data(corrected_targets_data, os.path.join(OUTPUT_PATH, 'data_corrected_target_pts.pkl'))
