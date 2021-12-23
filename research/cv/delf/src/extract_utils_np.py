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
"""utils to extract features"""
import numpy as np

def CalculateKeypointCenters(boxes):
    """Helper function to compute feature centers, from RF boxes.

    Args:
    boxes: [N, 4] float array.

    Returns:
    centers: [N, 2] float array.
    """
    return (boxes[:, (0, 1)] + boxes[:, (2, 3)])/2.0

def ApplyPcaAndWhitening(data,
                         pca_matrix,
                         pca_mean,
                         output_dim,
                         use_whitening=False,
                         pca_variances=None):
    """Applies PCA/whitening to data.

    Args:
    data: [N, dim] float array containing data which undergoes PCA/whitening.
    pca_matrix: [dim, dim] float array PCA matrix, row-major.
    pca_mean: [dim] float array, mean to subtract before projection.
    output_dim: Number of dimensions to use in output data, of type int.
    use_whitening: Whether whitening is to be used.
    pca_variances: [dim] float array containing PCA variances. Only used if
        use_whitening is True.

    Returns:
    output: [N, output_dim] float array with output of PCA/whitening operation.
    """
    b = np.transpose(pca_matrix[:output_dim, :data.shape[1]], (1, 0))
    output = np.matmul((data - pca_mean), b)

    # Apply whitening if desired.
    if use_whitening:
        output = output / np.sqrt(pca_variances[:output_dim])

    return output

def PostProcessDescriptors(descriptors, use_pca, pca_parameters=None):
    """Post-process descriptors.

    Args:
    descriptors: [N, input_dim] float array.
    use_pca: Whether to use PCA.
    pca_parameters: Only used if `use_pca` is True. Dict containing PCA
        parameter tensors, with keys 'mean', 'matrix', 'dim', 'use_whitening',
        'variances'.

    Returns:
    final_descriptors: [N, output_dim] float array with descriptors after
        normalization and (possibly) PCA/whitening.
    """
    # L2-normalize, and if desired apply PCA (followed by L2-normalization).
    final_descriptors = descriptors / np.linalg.norm(x=descriptors, ord=2, axis=1, keepdims=True)

    if use_pca:
        # Apply PCA, and whitening if desired.
        final_descriptors = ApplyPcaAndWhitening(final_descriptors,
                                                 pca_parameters['matrix'],
                                                 pca_parameters['mean'],
                                                 pca_parameters['dim'],
                                                 pca_parameters['use_whitening'],
                                                 pca_parameters['variances'])

    # Re-normalize.
    final_descriptors = final_descriptors / np.linalg.norm(x=final_descriptors, ord=2, axis=1, keepdims=True)

    return final_descriptors

def DelfFeaturePostProcessing(boxes, descriptors, use_pca, pca_parameters=None):
    """Extract DELF features from input image.

    Args:
    boxes: [N, 4] float array which denotes the selected receptive box. N is
        the number of final feature points which pass through keypoint selection
        and NMS steps.
    descriptors: [N, input_dim] float array.
    use_pca: Whether to use PCA.
    pca_parameters: Only used if `use_pca` is True. Dict containing PCA
        parameter tensors, with keys 'mean', 'matrix', 'dim', 'use_whitening',
        'variances'.

    Returns:
    locations: [N, 2] float array which denotes the selected keypoint
        locations.
    final_descriptors: [N, output_dim] float array with DELF descriptors after
        normalization and (possibly) PCA/whitening.
    """

    # Get center of descriptor boxes, corresponding to feature locations.
    locations = CalculateKeypointCenters(boxes)
    final_descriptors = PostProcessDescriptors(descriptors, use_pca,
                                               pca_parameters)

    return locations, final_descriptors
