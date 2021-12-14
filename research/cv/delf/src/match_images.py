# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""
import os
import argparse

import numpy as np
from scipy import spatial
from skimage import feature
from skimage import measure
from skimage import transform
import feature_io
import matplotlib
# Needed before pyplot import for matplotlib to work properly.
matplotlib.use('Agg')
import matplotlib.image as mpimg    #pylint: disable=wrong-import-position
import matplotlib.pyplot as plt     #pylint: disable=wrong-import-position

_DISTANCE_THRESHOLD = 0.8

parser = argparse.ArgumentParser()
parser.register('type', 'bool', lambda v: v.lower() == 'true')

parser.add_argument('--list_images_path', type=str, default="list_images.txt")
parser.add_argument('--images_path', type=str, default="")
parser.add_argument('--feature_path', type=str, default="")
parser.add_argument(
    '--output_image',
    type=str,
    default='test_match.png',
    help="""
    Path where an image showing the matches will be saved.
    """)
args = parser.parse_known_args()[0]

def ReadImageList(list_path):
    f = open(list_path, "r")
    image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths

def main():
    image_paths = ReadImageList(args.list_images_path)
    image_1_path = os.path.join(args.images_path, image_paths[0])+'.jpg'
    image_2_path = os.path.join(args.images_path, image_paths[1])+'.jpg'

    features_1_path = os.path.join(args.feature_path, image_paths[0])
    features_2_path = os.path.join(args.feature_path, image_paths[1])

    # Read features.
    locations_1, _, descriptors_1, _ = feature_io.ReadFromFile(
        features_1_path)
    num_features_1 = locations_1.shape[0]
    print(f"Loaded image 1's {num_features_1} features")
    locations_2, _, descriptors_2, _ = feature_io.ReadFromFile(
        features_2_path)
    num_features_2 = locations_2.shape[0]
    print(f"Loaded image 2's {num_features_2} features")

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = spatial.cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    # Perform geometric verification using RANSAC.
    _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                transform.AffineTransform,
                                min_samples=3,
                                residual_threshold=20,
                                max_trials=1000)

    print(f'Found {sum(inliers)} inliers')

    # Visualize correspondences, and save to file.
    _, ax = plt.subplots()
    img_1 = mpimg.imread(image_1_path)
    img_2 = mpimg.imread(image_2_path)
    inlier_idxs = np.nonzero(inliers)[0]
    feature.plot_matches(
        ax,
        img_1,
        img_2,
        locations_1_to_use,
        locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        matches_color='b')
    ax.axis('off')
    ax.set_title('DELF correspondences')
    plt.savefig(args.output_image)


if __name__ == '__main__':
    main()
