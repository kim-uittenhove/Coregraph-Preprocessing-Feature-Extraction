import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from utils import save_pickle_data, load_pickle_data

def plot_dataset(df, condition):
    plt.figure(figsize=(10, 6))
    plt.title(f'Dataset: {df.id[0]}, Condition: {condition}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(df['X'], df['Y'], color='grey', s=1, label='All Points')
    pendown_points = df[df['Pressure'] > 0]
    plt.scatter(pendown_points['X'], pendown_points['Y'], color='black', s=1, label='Pendown Points')
    plt.legend()
    plt.show()

def calibrate_time(df, pendown):
    t_start = pendown.Time.iloc[0]
    t_end = pendown.Time.iloc[-1]
    df.Time = df.Time - t_start
    return t_start, t_end

def compute_lifts(df, t_start, t_end):
    """
    Computes when the pen is lifted based on the pressure data and time calibration.
    
    Parameters:
        df (DataFrame): The DataFrame to compute lifts on.
        t_start (float): Start time of pendown.
        t_end (float): End time of pendown.
        
    Returns:
        DataFrame: Updated DataFrame with lift information.
    """
    df['lift'] = df['Pressure'].apply(lambda x: 1 if x == 0 else 0)
    df.loc[0, 'lift'] = -1
    for i in range(1, len(df)):
        if df.loc[i, 'lift'] == 1 and (df.loc[i, 'Time'] < t_start or df.loc[i, 'Time'] > t_end):
            df.loc[i, 'lift'] = -1
    return df

def set_nan_for_no_pressure(df):
    """
    Sets X and Y coordinates to NaN wherever the pressure is zero.
    
    Parameters:
        df (DataFrame): DataFrame to modify.
        
    Returns:
        DataFrame: Modified DataFrame with NaNs set for no pressure points.
    """
    df.loc[df['Pressure'] == 0, ['X', 'Y']] = np.nan
    return df


def smooth_data(df, x_col, y_col, window_length, polyorder):
    """
    Applies Savitzky-Golay smoothing to the specified x and y columns.
    
    Parameters:
        df (DataFrame): DataFrame containing the data.
        x_col (str): Column name for x data.
        y_col (str): Column name for y data.
        window_length (int): Length of the filter window (number of coefficients).
        polyorder (int): Order of the polynomial used to fit the samples.
        
    Returns:
        DataFrame: DataFrame with additional smoothed columns.
    """
    df[x_col] = df[x_col].interpolate(method='linear', limit_direction='both')
    df[y_col] = df[y_col].interpolate(method='linear', limit_direction='both')

    # Applying Savitzky-Golay filter to the interpolated columns
    df[f"{x_col}_smooth"] = savgol_filter(df[x_col], window_length, polyorder)
    df[f"{y_col}_smooth"] = savgol_filter(df[y_col], window_length, polyorder)
    return df


def compute_distance(df):
    """
    Computes the distance between consecutive points based on smoothed coordinates.
    
    Parameters:
        df (DataFrame): DataFrame containing smoothed position data.
        
    Returns:
        DataFrame: Updated DataFrame with distance computed.
    """
    x_diff = np.diff(df['X_smooth'], prepend=df['X_smooth'][0])
    y_diff = np.diff(df['Y_smooth'], prepend=df['Y_smooth'][0])
    df['distance'] = np.sqrt(x_diff**2 + y_diff**2)
    return df

def compute_velocity_and_acceleration(df):
    """
    Computes the velocity, acceleration, and jerk for the data points in the DataFrame.
    Velocity is calculated as the first derivative of the smoothed position data with respect to time,
    acceleration as the derivative of velocity, and jerk as the derivative of acceleration.

    The np.gradient function is used to compute the numerical derivative. 

    Parameters:
        df (DataFrame): DataFrame to compute these metrics for. The DataFrame must include 'X_smooth' and 'Y_smooth' 
        columns, which contain the smoothed position data of an object.

    Returns:
        DataFrame: Updated DataFrame with 'velocity_x', 'velocity_y', 'velocity', 'acceleration', and 'jerk' columns added.
        These columns represent the x-component of velocity, y-component of velocity, total velocity magnitude, 
        total acceleration, and jerk magnitude respectively.

    Note:
        - Velocity is calculated in units corresponding to the units of position divided by the units of time. In our case, 
        velocity units are pixels/second.
        - dt (delta t) is the time step between each position measurement, critical for determining the rate of change.
    """
    dt = 5 / 1000.0  # Time step in seconds, assuming measurements are taken every 5 milliseconds
    df['velocity_x'] = np.gradient(df['X_smooth'], dt)  # Compute the derivative of X position to get X component of velocity
    df['velocity_y'] = np.gradient(df['Y_smooth'], dt)  # Compute the derivative of Y position to get Y component of velocity
    df['velocity'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)  # Calculate magnitude of velocity vector
    df['acceleration'] = np.gradient(df['velocity'], dt)  # Compute the derivative of velocity to get acceleration
    df['jerk'] = np.gradient(df['acceleration'], dt)  # Compute the derivative of acceleration to get jerk

    return df

def spatial_resample(df, rate):
    """
    Resamples spatial data, based on a specified rate (=distance), by calculating cumulative distances.
    Resampled points are evenly distributed in space rather than in time.
    
    Parameters:
        df (DataFrame): DataFrame to resample.
        rate (float): Distance threshold for resampling.
        
    Returns:
        DataFrame: Updated DataFrame with resampling indicators.
    """
    var = 'resample'
    df[var] = 0
    dx = df['X_smooth'].diff()
    dy = df['Y_smooth'].diff()
    distances = np.sqrt(dx**2 + dy**2)
    cumulative_distance = distances.cumsum()
    sample_points = cumulative_distance // rate
    df[var] = (sample_points.diff() > 0) & (df['Pressure'] > 0)
    return df

def direction(df):
    """    
    The direction is computed by taking the difference between consecutive x and y coordinates of the resampled
    points, and normalizing these differences to create unit vectors. This normalization helps in standardizing
    the direction vectors regardless of the distance between points.

    Parameters:
        df (DataFrame): DataFrame containing the data with columns 'X_smooth' and 'Y_smooth' for the smoothed
        x and y coordinates, and a 'resample' column that indicates whether a point should be used in
        direction calculations.

    Returns:
        DataFrame: The updated DataFrame includes two new columns, 'dir_h' for horizontal (x) and 'dir_v'
        for vertical (y) components of the direction vectors. NaN indicates points where direction could not be computed 
        due to division by zero (when consecutive points are identical).

    Notes:
        - The function calculates differences using pandas' `diff()` method which subtracts the previous point from the
          current point in the sequence. It skips the first of the resampled points because there is no previous point to
          compare.
        - The 'resample' column identifies the points for analysis.
    """
    sampled_indices = df[df['resample'] == 1].index
    df['dir_h'] = np.nan
    df['dir_v'] = np.nan
    dx = df.loc[sampled_indices, 'X_smooth'].diff().iloc[1:]
    dy = df.loc[sampled_indices, 'Y_smooth'].diff().iloc[1:]
    norm = np.sqrt(dx**2 + dy**2)
    dir_h = np.where(norm != 0, dx / norm, np.nan)
    dir_v = np.where(norm != 0, dy / norm, np.nan)
    df.loc[sampled_indices[1:], 'dir_h'] = dir_h
    df.loc[sampled_indices[1:], 'dir_v'] = dir_v
    return df


def process_data(input_data, window_length, polyorder):
    """
    Processes input datasets by calibrating time, computing lifts, smoothing data,
    and computing metrics like distance, velocity, acceleration, jerk, and directions.
    
    Parameters:
        input_data (dict): Dictionary of datasets categorized by conditions.
        window_length (int): Window length for Savitzky-Golay filter.
        polyorder (int): Polynomial order for Savitzky-Golay filter.
        
    Returns:
        dict: Dictionary of processed datasets.
    """
    processed_data = {}
    for key, dataset in input_data.items():
        processed_dataset = []
        for df in dataset:
            df.Time = list(range(1, len(df.Time) + 1))
            df.Time = df.Time * 5 - 5
            df = set_nan_for_no_pressure(df)
            pendown = df[df.Pressure > 0]
            if pendown.empty:
                print(f"Warning: Empty 'pendown' dataset for key '{key}'. Skipping...")
                continue
            t_start, t_end = calibrate_time(df, pendown)
            df = compute_lifts(df, t_start, t_end)
            df = smooth_data(df, 'X', 'Y', window_length, polyorder)
            df = compute_distance(df)
            df = compute_velocity_and_acceleration(df)
            df = spatial_resample(df, 50)
            df = direction(df)
            processed_dataset.append(df)
        processed_data[key] = processed_dataset
    return processed_data

# Replace OUTPUT_PATH with the actual path to your pickled files
OUTPUT_PATH = ''

pickle_file_names = ['tmt_a.pkl', 'tmt_a_long.pkl', 'tmt_b.pkl', 'tmt_b_long.pkl']
loaded_data = {}

for file_name in pickle_file_names:
    marker = file_name.split('.')[0]
    data = load_pickle_data(os.path.join(OUTPUT_PATH, file_name))
    
    # Check if data is empty
    if not data:
        print(f"Warning: {file_name} contains empty data. Skipping...")
        continue
    
    loaded_data[marker] = data

window_length = 7
polyorder = 3

processed_data = process_data(loaded_data, window_length, polyorder)

# Save the processed_data as a single pickled file
processed_pickle_file_name = 'preprocessed_data.pkl'
save_pickle_data(processed_data, os.path.join(OUTPUT_PATH, processed_pickle_file_name))
