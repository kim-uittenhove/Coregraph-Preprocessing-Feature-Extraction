# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:24:36 2023

@author: kim uittenhove
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
from utils import save_pickle_data, load_pickle_data
from matplotlib import rcParams

def find_plateaus(data, tolerance, min_length):
    """
    Identifies plateaus in the data where values stay within a certain range.
    These plateaus would not be detected when detecting peaks
    Identify the midpoints of these plateaus 
    
    Parameters:
        data (np.array): Array of data points.
        tolerance (float): The maximum difference between values considered to be on the same plateau.
        min_length (int): Minimum number of consecutive data points required to form a plateau.
        
    Returns:
        list of tuples: List containing start and end indices of each detected plateau.
    """
    plateaus = []
    start = None
    for i in range(1, len(data)):
        if start is not None:
            if abs(data[i] - data[start]) > tolerance or i == len(data) - 1:
                if i - start >= min_length and (start == 0 or data[start-1] > data[start]) and data[i] > data[start]:
                    plateaus.append((start, i-1))
                start = None
        elif abs(data[i] - data[i-1]) <= tolerance:
            start = i - 1
    return plateaus


def detect_turning_points_velo(df, widths=np.arange(1, 100), include_plateaus=True):
    """
    Detects turning points in velocity data based on peaks and optional plateaus.
    Consider only the section of the data where the pen is down: df.lift == 0
    Find local minima in velocity by finding peaks in the inverted velocity values

    Parameters:
        df (DataFrame): Pandas DataFrame containing the velocity and other data.
        widths (np.array): Widths to use for the peak detection algorithm.
        include_plateaus (bool): If True, include plateaus in the detection of turning points.

    Returns:
        DataFrame: Modified DataFrame with a new column indicating turning points.
    """
    df['turnpoint'] = 0
    df_lift_0 = df[df['lift'] == 0]
    inverted_velocity = -1 * df_lift_0['velocity'].values
    local_minima_indices = find_peaks_cwt(inverted_velocity, widths, min_snr = 1)

    turning_point_indices = local_minima_indices
    if include_plateaus:
        plateau_indices = find_plateaus(df_lift_0['velocity'].values, tolerance=100, min_length=10)
        plateau_indices_flattened = [int((start + end) / 2) for start, end in plateau_indices]
        turning_point_indices = np.unique(np.concatenate([turning_point_indices, plateau_indices_flattened]))

    for index in turning_point_indices:
        original_index = df_lift_0.index[index]
        df.loc[original_index, 'turnpoint'] = 1

    return df

def plot_path_with_turning_points(df):
    """
    Plots the movement path with marked turning points.
    
    Parameters:
        df (DataFrame): DataFrame containing the x and y coordinates and turnpoint indicators.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(df['X_smooth'], df['Y_smooth'], label='Path', s = 0.5, c = "black")
    plt.scatter(df[df['turnpoint'] == 1]['X_smooth'], df[df['turnpoint'] == 1]['Y_smooth'], color='red', marker='x', label='Turnpoints')
    plt.xlabel('X_smooth')
    plt.ylabel('Y_smooth')
    plt.legend()
    plt.title(df['id'][0])
    plt.show()
    
    
def plot_velocity(df, condition):

    # Set font properties globally
    rcParams['font.family'] = 'serif' 
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 12  

    filtered_df = df[df['Pressure'] > 0]
    
    # Convert time from milliseconds to seconds for plotting
    filtered_df['Time_seconds'] = filtered_df['Time'] / 1000.0

    plt.figure(figsize=(10, 5))
    # Plot velocity as a black line
    plt.plot(filtered_df['Time_seconds'], filtered_df['velocity'], 'k-')  # 'k-' specifies a black line
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (pixels/s)')
    
    plt.title(f"Subject {df['id'].iloc[0]} - {condition}", loc='right', fontsize=12)

    # Customize axes to remove frame except for the bottom and left lines
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # Ensure directory exists for saving the plot
    base_dir = 'velocity'
    subject_dir = os.path.join(base_dir, str(df['id'].iloc[0]))
    os.makedirs(subject_dir, exist_ok=True)

    # Save plot to file 
    plot_file_name = f"velocity_{df['id'].iloc[0]}_{condition}.png"
    plot_path = os.path.join(subject_dir, plot_file_name)
    plt.savefig(plot_path, dpi=300)  # Increase resolution by setting DPI
    plt.close()


def process_data(input_data):
    """
    Processes each dataset by detecting turning points and generating plots.
    
    Parameters:
        input_data (dict): Dictionary of datasets categorized by conditions.
        
    Returns:
        dict: Dictionary of processed datasets.
    """
    processed_data = {}
    for condition, dataset in input_data.items():
        processed_dataset = []
        for df in dataset:
            df = detect_turning_points_velo(df, include_plateaus=False)            
            plot_path_with_turning_points(df)   
            plot_velocity(df, condition)
            processed_dataset.append(df)
        processed_data[condition] = processed_dataset
    return processed_data

# Load and process data
OUTPUT_PATH = ''
loaded_data = load_pickle_data(os.path.join(OUTPUT_PATH, 'preprocessed_data.pkl'))
data_turning_pts = process_data(loaded_data)
save_pickle_data(data_turning_pts, os.path.join(OUTPUT_PATH, 'processed_data_points_velocity.pkl'))

