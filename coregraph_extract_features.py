# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:58:38 2024

This script processes trajectory data to extract line and target features, calculate summary measures,
plot velocity profiles, and save the processed features.

Author: kim uittenhove
"""

import os
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from utils import save_pickle_data, load_pickle_data

def extract_features(traj_data: pd.DataFrame, feature_type: str, condition: str = None):
    """
    Extracts features from either line or target sections in the input data.

    Parameters:
        traj_data: DataFrame containing trajectory data.
        feature_type (str): Either 'line' or 'target' to specify the type of features to extract.
        condition (str): Condition name for plotting (default is None).

    Returns:
        DataFrame: DataFrame containing extracted features. 
        or None
    """
    zone_col, pause_col, correct_col = get_column_names(feature_type)
    pendown = traj_data[(traj_data.Pressure > 0) & (traj_data[zone_col] > 0)]
    
    if pendown.empty:
        print(f"Warning: No {feature_type} sections found in dataframe with ID: {traj_data.id[0]}")
        return None

    features = []
    for section in pendown[zone_col].unique():
        section_data = pendown[pendown[zone_col] == section].copy()
        section_features = {'section': section, 'correct': section_data[correct_col].iloc[0]}
        
        if feature_type == 'line':
            section_features.update(calculate_kinematic_features(section_data))
            plot_velocity_profile(section_data, section_features, traj_data.id[0], section, condition)
            section_features.update(calculate_line_features(section_data, pause_col))
        else:
            section_features.update(calculate_target_features(section_data, pause_col))
        
        features.append(section_features)

    return pd.DataFrame(features).set_index('section') if features else None

def get_column_names(feature_type: str):
    """Get column names based on feature type."""
    zone_col = 'going_to_target' if feature_type == 'line' else 'target_zone'
    pause_col = 'pause_between' if feature_type == 'line' else 'pause_on'
    correct_col = 'line_connect_correct' if feature_type == 'line' else 'target_zone_correct'
    return zone_col, pause_col, correct_col

def calculate_line_features(section_data: pd.DataFrame, pause_col: str) -> Dict[str, float]:
    """Calculate line-specific features."""
    return {
        'pause_line': section_data[pause_col].sum() * 5,
        'avg_velocity_line': section_data['velocity'].mean(),
        'avg_pressure_line': section_data['Pressure'].mean(),
        'sd_pressure_line': section_data['Pressure'].std(),
        'avg_jerk_line': section_data['jerk'].mean(),
        'Amp': section_data['distance'].sum(),
    }

def calculate_target_features(section_data: pd.DataFrame, pause_col: str) -> Dict[str, float]:
    """Calculate target-specific features."""
    return {
        'avg_velocity_target': section_data['velocity'].mean(),
        'avg_pressure_target': section_data['Pressure'].mean(),
        'pause_target': section_data[pause_col].sum() * 5,
        'target_section_length': section_data['distance'].sum()
    }


def calculate_kinematic_features(section_data: pd.DataFrame, config: Dict[str, int] = None):
    """
    Calculates kinematic features for a given section of trajectory data.
    The Window size for velocity smoothing has an impact on detection of peaks as well as on the measurement of maximum velocity.

    Parameters:
        section_data: DataFrame containing section of trajectory data.
        window_size (int): Window size for smoothing the velocity data.
        height_threshold (int): Minimum height of peaks.
        prominence_threshold (int): Minimum prominence of peaks.

    Returns:
        dict: Dictionary containing calculated kinematic features.
    """
    if config is None:
        config = {'window_size': 20, 'height_threshold': 5000, 'prominence_threshold': 500}
    
    section_data['smoothed_velocity'] = uniform_filter1d(section_data['velocity'], size=config['window_size'])
    peaks, properties = find_peaks(section_data['smoothed_velocity'], height=config['height_threshold'], prominence=config['prominence_threshold'])
    section_data.reset_index(drop=True, inplace=True)
    
    if not peaks.size:
        return get_default_kinematic_features()
    
    section_data['adjusted_time'] = section_data['Time'] - section_data['Time'].min()
    vmax_index = section_data['smoothed_velocity'].idxmax()
    vmax = section_data.at[vmax_index, 'smoothed_velocity']
    vmax_time = section_data.at[vmax_index, 'adjusted_time']
    
    movement_time = section_data['adjusted_time'].iloc[-1] / 1000
    amplitude = section_data['distance'].sum()
    dist_optimal = calculate_optimal_distance(section_data)
    path_efficiency = dist_optimal / amplitude
    
    threshold_times = calculate_threshold_times(section_data, vmax, vmax_index, vmax_time)
    threshold_features = calculate_threshold_features(section_data, threshold_times, vmax_time)
    
    return {
        'Vmax': vmax, 'peaks': peaks, 
        'TM': movement_time, 'Amp': amplitude, 'dist_optimal': dist_optimal, 'path_efficiency': path_efficiency,
        'Vmax_index': vmax_index, 'N_Peaks': len(peaks), 'peak_prominences': properties['prominences'],
        **threshold_features
    }

def get_default_kinematic_features():
    """Get default kinematic features when no peaks are found."""
    return {
        'Vmax': np.nan, 'peaks': np.array([]), 'TM': np.nan, 'Amp': np.nan, 'dist_optimal': np.nan, 'path_efficiency': np.nan,
        'Vmax_index': 0, 'N_Peaks': 0, 'peak_prominences': np.array([]),
        **{f'{prefix}_{int(t*100)}_{pos}': np.nan for prefix in ['TTP', 'Index_TTP', 'TimeDiff'] for t in [0.2, 0.4, 0.6, 0.8] for pos in ['before', 'after']},
        **{f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_{pos}': np.nan for t1, t2 in [(0.8, 0.6), (0.6, 0.4), (0.4, 0.2)] for pos in ['before', 'after']},
        **{f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_ratio': np.nan for t1, t2 in [(0.8, 0.6), (0.6, 0.4), (0.4, 0.2)]}
    }

def calculate_optimal_distance(section_data: pd.DataFrame):
    """Calculate the optimal distance between points."""
    return np.sqrt(
        (section_data.at[section_data.index[-1], 'X_smooth'] - section_data.at[section_data.index[0], 'X_smooth'])**2 +
        (section_data.at[section_data.index[-1], 'Y_smooth'] - section_data.at[section_data.index[0], 'Y_smooth'])**2
    )

def calculate_threshold_times(section_data: pd.DataFrame, vmax: float, vmax_index: int, vmax_time: float):
    """Calculate threshold times before and after the velocity peak."""
    threshold_times = {'before': {}, 'after': {}}
    
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        target_value = vmax * threshold
        index_before = find_lower_closest_index(section_data['smoothed_velocity'], target_value, 0, vmax_index, find_before=True)
        index_after = find_lower_closest_index(section_data['smoothed_velocity'], target_value, vmax_index, len(section_data), find_before=False)
        
        for pos, index in [('before', index_before), ('after', index_after)]:
            if not np.isnan(index):
                threshold_times[pos][threshold] = section_data.at[index, 'adjusted_time']
    
    return threshold_times

def find_lower_closest_index(velocity_array: pd.Series, value: float, start: int, end: int, find_before: bool = True):
    """Find the index of the closest value below the target value."""
    segment = velocity_array[start:end]
    lower_indices = segment[segment < value].index
    if lower_indices.empty:
        return np.nan
    return lower_indices[-1] if find_before else lower_indices[0]

def calculate_threshold_features(section_data: pd.DataFrame, threshold_times: Dict[str, Dict[float, float]], vmax_time: float):
    """Calculate threshold-based features.
    This includes:
        - the time between the threshold and the vmax peak 
        - the time between successive tresholds
        - The ratio of these features, where features before the peak are divided by features after the peak
            # Values < 1 indicate a relatively shorter/steeper acceleration phase 
            # Values > 1 indicate a relatively shorter/steeper deceleration phase
    """
    features = {}
    
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        for pos in ['before', 'after']:
            if threshold in threshold_times[pos]:
                time = threshold_times[pos][threshold]
                features[f'TTP_{int(threshold*100)}_{pos}'] = time
                features[f'TimeDiff_{int(threshold*100)}_{pos}'] = np.abs(time - vmax_time)
                
                if pos == 'before':
                    index = find_lower_closest_index(section_data['smoothed_velocity'], section_data.at[section_data['smoothed_velocity'].idxmax(), 'smoothed_velocity'] * threshold, 0, section_data['smoothed_velocity'].idxmax(), find_before=True)
                else:
                    index = find_lower_closest_index(section_data['smoothed_velocity'], section_data.at[section_data['smoothed_velocity'].idxmax(), 'smoothed_velocity'] * threshold, section_data['smoothed_velocity'].idxmax(), len(section_data), find_before=False)
                
                features[f'Index_TTP_{int(threshold*100)}_{pos}'] = index
            else:
                features[f'TTP_{int(threshold*100)}_{pos}'] = np.nan
                features[f'TimeDiff_{int(threshold*100)}_{pos}'] = np.nan
                features[f'Index_TTP_{int(threshold*100)}_{pos}'] = np.nan
        
        before_key = f'TimeDiff_{int(threshold*100)}_before'
        after_key = f'TimeDiff_{int(threshold*100)}_after'
        if not np.isnan(features[before_key]) and not np.isnan(features[after_key]) and features[after_key] != 0:
            features[f'TimeDiff_{int(threshold*100)}_ratio'] = features[before_key] / features[after_key]
        else:
            features[f'TimeDiff_{int(threshold*100)}_ratio'] = np.nan
    
    for t1, t2 in [(0.8, 0.6), (0.6, 0.4), (0.4, 0.2)]:
        for pos in ['before', 'after']:
            if t1 in threshold_times[pos] and t2 in threshold_times[pos]:
                features[f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_{pos}'] = np.abs(threshold_times[pos][t1] - threshold_times[pos][t2])
            else:
                features[f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_{pos}'] = np.nan
        
        before_key = f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_before'
        after_key = f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_after'
        if not np.isnan(features[before_key]) and not np.isnan(features[after_key]) and features[after_key] != 0:
            features[f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_ratio'] = features[before_key] / features[after_key]
        else:
            features[f'TimeDiff_{int(t1*100)}_to_{int(t2*100)}_ratio'] = np.nan
    
    return features

def plot_velocity_profile(section_data: pd.DataFrame, features: Dict[str, Union[float, np.ndarray]], subject_id: str, section: int, condition: str):
    """Plot the velocity profile of a given section and save the plot."""
    plot_dir = os.path.join('kinematics', str(subject_id))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"movement_{section}_{condition}.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(section_data['Time'], section_data['smoothed_velocity'], label='Smoothed Velocity', color='blue')
    
    peak_times = section_data.Time.iloc[features.get('peaks', [])]
    peak_velocities = section_data.smoothed_velocity.iloc[features.get('peaks', [])]
    
    if peak_times.size > 0:
        ax.plot(peak_times, peak_velocities, 'x', color='orange', label='Peaks')
    
    vmax_index = features['Vmax_index']
    ax.scatter(section_data.loc[vmax_index, 'Time'], section_data.loc[vmax_index, 'smoothed_velocity'], marker='d', color='red', s=100, label='Vmax')
    
    if peak_times.size > 0:
        ax.axvline(x=peak_times.iloc[0], color='r', linestyle='--', label='First Peak')
        ax.axvline(x=peak_times.iloc[-1], color='g', linestyle='--', label='Last Peak')
    
    for threshold in [20, 40, 60, 80]:
        plot_threshold_point(ax, section_data, features, threshold, 'before', 'bo')
        plot_threshold_point(ax, section_data, features, threshold, 'after', 'bo')

    ax.set_title(f"Velocity Profile with AT and DT for Section {section}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Smoothed Velocity')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

def plot_threshold_point(ax: plt.Axes, section_data: pd.DataFrame, features: Dict[str, Union[float, np.ndarray]], threshold: int, position: str, marker: str):
    """Plot a threshold point on the velocity profile."""
    index_key = f'Index_TTP_{threshold}_{position}'
    time_diff_key = f'TimeDiff_{threshold}_{position}'

    if index_key in features and not np.isnan(features[index_key]):
        index = int(features[index_key])
        ax.plot(section_data['Time'].iloc[index], section_data['smoothed_velocity'].iloc[index], marker)
        ax.annotate(f'{threshold}%\n{features[time_diff_key]:.0f}',
                    (section_data['Time'].iloc[index], section_data['smoothed_velocity'].iloc[index]),
                    textcoords="offset points", xytext=(0, 10), ha='center', color='black')

def calculate_summary_measures(line_features_df: pd.DataFrame, target_features_df: pd.DataFrame) -> Dict:
    """
    Calculates summary measures from line and target features.

    Parameters:
        line_features_df (DataFrame): DataFrame containing line features.
        target_features_df (DataFrame): DataFrame containing target features.

    Returns:
        dict: Dictionary containing summary measures.
    """
    if line_features_df is None or 'correct' not in line_features_df.columns:
        print("Error: 'correct' column not found in line_features_df")
        return None

    if target_features_df is None or 'correct' not in target_features_df.columns:
        print("Error: 'correct' column not found in target_features_df")
        return None

    results = {}
    correct_lines = line_features_df[line_features_df['correct'] == 1]
    correct_targets = target_features_df[target_features_df['correct'] == 1]
    line_averages = calculate_means(correct_lines, ['pause_line', 'avg_velocity_line', 'avg_pressure_line', 'Amp', 'path_efficiency', 'sd_pressure_line'])
    target_averages = calculate_means(correct_targets, ['pause_target', 'avg_velocity_target', 'avg_pressure_target', 'target_section_length'])
    
    one_peak_lines = correct_lines[correct_lines['N_Peaks'] == 1]
    threshold_ratio_values, threshold_ratio_averages, avg_threshold_ratio = calculate_threshold_ratios(one_peak_lines)
    threshold_diff_ratio_values, threshold_diff_ratio_averages, avg_threshold_diff_ratio = calculate_threshold_diff_ratios(one_peak_lines)
    
    results.update({
        **line_averages,
        'one_peak_N': len(one_peak_lines),
        'one_peak_vmax': one_peak_lines['Vmax'].mean(skipna=True),
        'one_peak_jerk': one_peak_lines['avg_jerk_line'].mean(skipna=True),
        'one_peak_avg_dist_time_ratio': (one_peak_lines['Amp'] / one_peak_lines['TM']).mean(skipna=True),
        'avg_N_Peaks': correct_lines[correct_lines['N_Peaks'] >= 1]['N_Peaks'].mean(skipna=True),
        'count_correct_lines': correct_lines.shape[0],
        **threshold_ratio_averages,
        'threshold_ratio_values': threshold_ratio_values,
        **threshold_diff_ratio_averages,
        'threshold_diff_ratio_values': threshold_diff_ratio_values,
        'avg_threshold_to_peak_ratio': avg_threshold_ratio,
        'avg_threshold_slopes_ratio': avg_threshold_diff_ratio,
        **target_averages,
        'count_correct_targets': correct_targets.shape[0],
        'ratio_target_line': target_averages['avg_target_section_length'] / line_averages['avg_Amp'] if line_features_df is not None else None
    })

    return results

def calculate_means(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """Calculate means for the specified columns, ignoring NaNs."""
    return {f'avg_{col}': df[col].mean(skipna=True) for col in columns}

def calculate_threshold_ratios(one_peak_lines: pd.DataFrame) -> Tuple[Dict[str, List[float]], Dict[str, float], float]:
    """Calculate threshold ratio values and averages, ignoring NaNs."""
    threshold_ratio_values = {f'TimeDiff_{threshold}_ratio': one_peak_lines[f'TimeDiff_{threshold}_ratio'].tolist() for threshold in [20, 40, 60, 80]}
    threshold_ratio_averages = {f'avg_TimeDiff_{threshold}_ratio': one_peak_lines[f'TimeDiff_{threshold}_ratio'].mean(skipna=True) for threshold in [20, 40, 60, 80]}
    avg_threshold_ratio = pd.Series(threshold_ratio_averages).mean(skipna=True)
    return threshold_ratio_values, threshold_ratio_averages, avg_threshold_ratio

def calculate_threshold_diff_ratios(one_peak_lines: pd.DataFrame) -> Tuple[Dict[str, List[float]], Dict[str, float], float]:
    """Calculate threshold difference ratio values and averages, ignoring NaNs."""
    threshold_diff_ratio_values = {f'TimeDiff_{t1}_to_{t2}_ratio': one_peak_lines[f'TimeDiff_{t1}_to_{t2}_ratio'].tolist() for t1, t2 in [(80, 60), (60, 40), (40, 20)]}
    threshold_diff_ratio_averages = {f'avg_TimeDiff_{t1}_to_{t2}_ratio': one_peak_lines[f'TimeDiff_{t1}_to_{t2}_ratio'].mean(skipna=True) for t1, t2 in [(80, 60), (60, 40), (40, 20)]}
    avg_threshold_diff_ratio = pd.Series(threshold_diff_ratio_averages).mean(skipna=True)
    return threshold_diff_ratio_values, threshold_diff_ratio_averages, avg_threshold_diff_ratio

def process_data(corrected_data: Dict[str, List[pd.DataFrame]], keys_to_process: List[str]):
    features_by_condition = {}

    for key in keys_to_process:
        if key in corrected_data:
            all_combined_measures = []
            for df in corrected_data[key]:
                line_features_df = extract_features(df, 'line', key)
                target_features_df = extract_features(df, 'target')
                section_features = calculate_summary_measures(line_features_df, target_features_df)
                if section_features:
                    task_T = df[df.Pressure > 0].Time.iloc[-1]
                    general_features = {
                        'ID': df.id.iloc[0],
                        'task_T': task_T,
                        'pretask_T': df.Time.iloc[0] if not df.empty else 0,
                        'lifts_N': df[(df.Time >= 0) & (df.Time <= task_T)].lift_enact.sum(),
                        'lifts_DUR': df[(df.Time >= 0) & (df.Time <= task_T)].lift.sum() * 5,
                        'correct': df[df.connect_correct == 1].shape[0]
                    }
                    all_combined_measures.append({**section_features, **general_features})
            features_by_condition[key] = all_combined_measures

    return features_by_condition

# Load the processed data
OUTPUT_PATH = ''
data = load_pickle_data(os.path.join(OUTPUT_PATH, 'data_lifts_pauses_zones.pkl'))

# Define conditions and process the data
keys_to_process = ['tmt_a', 'tmt_b', 'tmt_a_long', 'tmt_b_long']
features_by_condition = process_data(data, keys_to_process)

# Save the processed features
save_pickle_data(features_by_condition, os.path.join(OUTPUT_PATH, 'features_by_condition.pkl'))