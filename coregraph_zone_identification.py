# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:39:22 2023

@author: kim uittenhove
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import save_pickle_data, load_pickle_data


def add_target_zone_and_correct(df):
    """
    Adds 'target_zone' and 'target_zone_correct' columns to the dataframe.
    Identifies target zones and sets the 'target_zone' and 'target_zone_correct' for 
    rows around each identified target until a row with 'cluster' > -1 or not NaN is found, 
    while skipping rows where 'Time' is negative.

    Parameters:
        df (DataFrame): The input dataframe containing trajectory data.
    """
    df['target_zone'] = np.nan
    df['target_zone_correct'] = np.nan

    # Iterate over the DataFrame
    for index, row in df.iterrows():
        # Skip rows with negative 'Time'
        if row['Time'] < 0:
            continue
        
        # Find rows where 'target_id' > -1
        if row['target_id'] > -1:
            target_id = row['target_id']
            target_correct = row['connect_correct']

            # Update current row immediately
            df.at[index, 'target_zone'] = target_id
            df.at[index, 'target_zone_correct'] = target_correct

            # Update previous rows until a row with 'cluster' > -1 is found
            j = index - 1
            while j >= 0 and (pd.isna(df.at[j, 'cluster']) or df.at[j, 'cluster'] == -1):
                if df.at[j, 'Time'] < 0:  # Check for negative time before updating
                    break
                df.at[j, 'target_zone'] = target_id
                df.at[j, 'target_zone_correct'] = target_correct
                j -= 1

            # Update next rows until a row with 'cluster' > -1 is found
            j = index + 1
            while j < len(df) and (pd.isna(df.at[j, 'cluster']) or df.at[j, 'cluster'] == -1):
                if df.at[j, 'Time'] < 0:  # Check for negative time before updating
                    break
                df.at[j, 'target_zone'] = target_id
                df.at[j, 'target_zone_correct'] = target_correct
                j += 1

    return df
          
def add_line_connect_correct(df):
    """
    Adds 'line_connect_correct' and 'going_to_target' columns to the dataframe,
    to indicate which target a line is connecting to, and whether that is a correct connection.
    The function checks if the end of a line section connects correctly when the target zone starts.

    Parameters:
        df (DataFrame): The input dataframe containing trajectory data.
    """
    df['line_connect_correct'] = np.nan
    df['going_to_target'] = np.nan

    i = 0
    while i < len(df):
        # Identify the start of a new line section
        if not pd.isna(df.at[i, 'linepoint']):
            start_index = i

            # Move to the end of the line section when target zone starts
            while i < len(df) and pd.isna(df.at[i, 'target_zone']):
                i += 1

            # Determine the end index of the line section
            end_index = i
            if end_index < len(df):
                # Determine if the line is connecting correctly
                line_correct = df.at[end_index, 'target_zone_correct']
                going_to_target_id = df.at[end_index, 'target_zone']

                # Update the dataframe with the correct target connection info
                df.loc[start_index:end_index - 1, 'line_connect_correct'] = line_correct
                df.loc[start_index:end_index - 1, 'going_to_target'] = going_to_target_id
        i += 1

    return df


def lift_enact(df):
    """
    Adds a 'lift_enact' column to the dataframe, indicating the moments where the pen goes from being 
    on the tablet to being off the tablet.

    Parameters:
        df (DataFrame): The input dataframe containing trajectory data. Must include 'Pressure', 'Time', and 'lift' columns.

    Returns:
        DataFrame: The dataframe with the 'lift_enact' column added.
    """
    # Initialize 'lift_enact' column to 0
    df['lift_enact'] = 0

    # Identify the pendown time range
    pendown = df[df.Pressure > 0]
    if pendown.empty:
        return df

    t_start, t_end = pendown['Time'].iloc[0], pendown['Time'].iloc[-1]

    # Filter the task dataframe to the pendown period and calculate lift enact moments
    task = df[(df['Time'] > t_start) & (df['Time'] < t_end)]
    lift_enact_indices = task.index[task['lift'].diff() == 1]

    # Update 'lift_enact' column
    df.loc[lift_enact_indices, 'lift_enact'] = 1

    return df

def pause(df):
    """
    Adds 'pause_between' and 'pause_on' columns to the dataframe, indicating between-target and on-target pauses 
    based on velocity.
    
    The velocity for a pause is set to < 100px/s

    Parameters:
        df (DataFrame): The input dataframe containing trajectory data. Must include 'Pressure', 'Time', 'linepoint', and 'velocity' columns.

    Returns:
        DataFrame: The dataframe with 'pause_between' and 'pause_on' columns added.
    """
    # Initialize 'pause_between' and 'pause_on' columns to 0
    df['pause_between'] = 0
    df['pause_on'] = 0

    # Identify the pendown time range
    pendown = df[df.Pressure > 0]
    if pendown.empty:
        return df  # No pendown data, return the original dataframe

    t_start, t_end = pendown['Time'].iloc[0], pendown['Time'].iloc[-1]

    # Filter the task dataframe to the pendown period
    task = df[(df['Time'] > t_start) & (df['Time'] < t_end)]

    # Identify between-target pauses
    between_target_pauses = task[(task['linepoint'] > 0) & (task['velocity'] < 100)].index
    df.loc[between_target_pauses, 'pause_between'] = 1

    # Identify on-target pauses
    on_target_pauses = task[pd.isna(task['linepoint']) & (task['velocity'] < 100)].index
    df.loc[on_target_pauses, 'pause_on'] = 1

    return df

               
def plot_velocity_profiles(data, conditions):

    for condition in conditions:
        condition_data = data[condition]

        for df in condition_data:
            # Determine subject_id if column exists
            subject_id = df['subject_id'].iloc[0] if 'subject_id' in df.columns else 'Unknown'

            # Set up the plot
            plt.figure(figsize=(10, 6))
            plt.plot(df['Time'], df['velocity'], label='Velocity', color='blue')

            # Highlight lifts
            lift_points = df[df['lift_enact'] == 1]
            plt.scatter(lift_points['Time'], lift_points['velocity'], label='Lifts', color='red', marker='^', zorder=5)

            # Highlight pauses between targets
            pause_between_points = df[df['pause_between_enact'] == 1]
            plt.scatter(pause_between_points['Time'], pause_between_points['velocity'], label='Pauses Between Targets', color='green', marker='o', zorder=5)

            # Highlight pauses on targets
            pause_on_points = df[df['pause_on_enact'] == 1]
            plt.scatter(pause_on_points['Time'], pause_on_points['velocity'], label='Pauses On Targets', color='orange', marker='x', zorder=5)

            # Highlight target points where target_id > -1
            target_points = df[df['target_id'] > -1]
            plt.scatter(target_points['Time'], target_points['velocity'], label='Target Points', color='purple', marker='s', zorder=5)

            # Adding labels and title
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            plt.title(f'Velocity Profile for Subject {subject_id}, Condition: {condition}')
            plt.legend()
            plt.grid(True)
            plt.show()

def process_data(corrected_data, keys_to_process):
    processed_data = {}
    for key in keys_to_process:
        if key in corrected_data:
            processed_dataset = []
            
            for df in corrected_data[key]:
                               
                add_target_zone_and_correct(df)
                add_line_connect_correct(df)
                lift_enact(df) 
                pause(df)    
               
                processed_dataset.append(df)
                
            processed_data[key] = processed_dataset
                
    return processed_data

OUTPUT_PATH = ''
corrected_data = load_pickle_data(os.path.join(OUTPUT_PATH, 'data_corrected_target_pts.pkl'))
                         
keys_to_process = ['tmt_a', 'tmt_b']
data_lifts_pauses_targetzones = process_data(corrected_data, keys_to_process)

# Call the function to plot the velocity profiles
#plot_velocity_profiles(data_lifts_pauses_targetzones, keys_to_process)

#save_pickle_data(data_lifts_pauses_targetzones, os.path.join(OUTPUT_PATH, 'data_lifts_pauses_zones_YA.pkl'))

def plot_trajectory_sections(data, conditions):
    for condition in conditions:
        condition_data = data[condition]

        for df in condition_data:
            # Determine subject_id if column exists
            subject_id = df['id'].iloc[0] if 'id' in df.columns else 'Unknown'

            # Set up the plot
            plt.figure(figsize=(10, 6))
            
            # Plot the entire trajectory line
            plt.plot(df['X'], df['Y'], label='Trajectory', color='lightgrey', alpha=0.5)
            
            # Highlight line sections with different colors based on 'going_to_target'
            unique_targets = df['going_to_target'].dropna().unique()
            colors = plt.cm.get_cmap('tab20', len(unique_targets))
            
            for i, target_id in enumerate(unique_targets):
                line_points = df[df['going_to_target'] == target_id]
                plt.plot(line_points['X'], line_points['Y'], label=f'Going to Target {int(target_id)}', color=colors(i), zorder=5)
            
            # Highlight target zones
            target_points = df[~pd.isna(df['target_zone'])]
            plt.scatter(target_points['X'], target_points['Y'], label='Target Zones', color='purple', marker='s', zorder=5)
            
            # Highlight actual target points
            actual_target_points = df[df['target_id'] > -1]
            plt.scatter(actual_target_points['X'], actual_target_points['Y'], label='Target Points', color='red', marker='^', zorder=5)
            
            # Adding labels and title
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Trajectory Sections for Subject {subject_id}, Condition: {condition}')
            plt.legend()
            plt.grid(True)
            plt.gca().invert_yaxis()  # Invert y-axis to match the typical display of coordinates in tablet data
            plt.show()


#plot_trajectory_sections(data_lifts_pauses_targetzones, keys_to_process)

