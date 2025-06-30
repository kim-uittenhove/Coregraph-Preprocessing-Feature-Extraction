# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:29:23 2023

@author: kim uittenhove
"""

import os
import pandas as pd
import re
from utils import save_pickle_data

# Disable chained assignment warning
pd.options.mode.chained_assignment = None

# Set data and output paths
Group = ""
DATA_PATH = ''
OUTPUT_PATH = ''
TASK_TYPES = ["tmt_a", "tmt_a_long", "tmt_b", "tmt_b_long"]

# Generate pickle file names from task types with the group suffix
pickle_file_names = [f"{task_type}_{Group}.pkl" for task_type in TASK_TYPES]

# List all subject folders in the data path
subjects = os.listdir(DATA_PATH)

# Dictionary to hold data frames for each task type
task_files = {task: [] for task in TASK_TYPES}

# Define regex patterns for each task type
regex_patterns = {
    "tmt_a_long": re.compile(r"tmt\-?a[\s\-]?long", re.IGNORECASE),
    "tmt_a": re.compile(r"tmt\-?a(?![\s\-]?long)", re.IGNORECASE),
    "tmt_b_long": re.compile(r"tmt\-?b[\s\-]?long", re.IGNORECASE),
    "tmt_b": re.compile(r"tmt\-?b(?![\s\-]?long)", re.IGNORECASE)
}

# List all subject folders in the data path
subjects = os.listdir(DATA_PATH)

for subject in subjects:
    subject_path = os.path.join(DATA_PATH, subject)
    files = [f for f in os.listdir(subject_path) if not f.endswith('.png')]

    # Process each file in the subject's folder
    for file in files:
        # Check each task type for a regex match
        for task_type, pattern in regex_patterns.items():
            if pattern.search(file):
                file_path = os.path.join(subject_path, file)
                df = pd.read_csv(file_path)
                df['id'] = subject
                task_files[task_type].append(df)
                break  # Stop after the first match to avoid duplicate entries

# Save data for each task type using the predefined filenames
for task_type, file_name in zip(TASK_TYPES, pickle_file_names):
    save_pickle_data(task_files[task_type], os.path.join(OUTPUT_PATH, file_name))
