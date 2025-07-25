# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:23:38 2023

@author: kim uittenhove
"""

import cv2
import numpy as np
from PIL import Image
import os

def click_event(event, x, y, flags, param):
    global coords
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        print(f"Clicked at ({x}, {y})")

conditions = ['tmt_a', 'tmt_a_long', 'tmt_b', 'tmt_b_long']

path_to_images = ''

image_paths = {
    'tmt_a': os.path.join(path_to_images, 'tmt_a.jpg'),
    'tmt_b': os.path.join(path_to_images, 'tmt_b.jpg'),
    'tmt_a_long': os.path.join(path_to_images, 'tmt_a_long.jpg'),
    'tmt_b_long': os.path.join(path_to_images, 'tmt_b_long.jpg'),
}

all_target_centers = {}

for condition in conditions:
    img = cv2.imread(image_paths[condition])

    coords = []
    cv2.namedWindow(condition)
    cv2.setMouseCallback(condition, click_event)

    while True:
        cv2.imshow(condition, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    target_centers = np.array(coords)
    numbered_target_centers = [(i + 1, center) for i, center in enumerate(target_centers)]
    all_target_centers[condition] = numbered_target_centers

np.savez('target_centers.npz', **all_target_centers)


