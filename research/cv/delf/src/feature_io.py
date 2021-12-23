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
"""IO for features"""
import numpy as np

def ReadFromFile(file_path):
    """Helper function to load data from a DelfFeatures format in a file.

    Args:
    file_path: Path to file containing data.

    Returns:
    locations: [N, 2] float array which denotes the selected keypoint
        locations. N is the number of features.
    scales: [N] float array with feature scales.
    descriptors: [N, depth] float array with DELF descriptors.
    attention: [N] float array with attention scores.
    """
    locations = np.load(file_path+'.location'+'.npz')['arr_0']
    scales = np.load(file_path+'.scale'+'.npz')['arr_0']
    descriptors = np.load(file_path+'.feature'+'.npz')['arr_0']
    attention = np.load(file_path+'.score'+'.npz')['arr_0']
    return locations, scales, descriptors, attention

def WriteToFile(out_desc_fullpath, locations_out, feature_scales_out, descriptors_out, attention_out):
    np.savez(out_desc_fullpath+'.location', locations_out)
    np.savez(out_desc_fullpath+'.scale', feature_scales_out)
    np.savez(out_desc_fullpath+'.feature', descriptors_out)
    np.savez(out_desc_fullpath+'.score', attention_out)

def ReadFeature(file_path):
    descriptors = np.load(file_path+'.feature'+'.npz')['arr_0']
    locations = np.load(file_path+'.location'+'.npz')['arr_0']
    return descriptors, locations
