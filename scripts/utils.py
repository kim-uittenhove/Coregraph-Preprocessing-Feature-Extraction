# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:56:01 2024

@author: kimui
"""

import pickle

def load_pickle_data(file_name):
    """Load data from a pickle file."""
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle_data(data, file_name):
    """Save data to a pickle file."""
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

#Thresholds for determining whether a turnpoint is in reach of a theoretical target point
max_thresholds = {
    'tmt_a': (0.10, 0.15),
    'tmt_b': (0.10, 0.15),
    'tmt_a_long': (0.06, 0.09),
    'tmt_b_long': (0.06, 0.09)
}
