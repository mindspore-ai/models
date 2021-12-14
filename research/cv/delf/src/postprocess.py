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
"""post process for 310 inference"""
import argparse
import time
import os

import numpy as np

import extract_utils_np as extract_utils
import box_list_np as box_list
import box_list_ops_np as box_list_ops
import feature_io
import delf_config

_STATUS_CHECK_ITERATIONS = 10

parser = argparse.ArgumentParser(description='MindSpore delf Example')

parser.add_argument('--use_list_txt', type=str,
                    default="False", choices=['True', 'False'])
parser.add_argument('--list_images_path', type=str, default="list_images.txt")

parser.add_argument('--bin_path', type=str, default="")
parser.add_argument('--size_path', type=str, default="")
parser.add_argument('--images_path', type=str, default="")
parser.add_argument('--target_path', type=str, default="")

# pca config
parser.add_argument('--use_pca', type=bool, default=True)
parser.add_argument('--mean_path', type=str, default="./pca/mean.npy")
parser.add_argument('--projection_matrix_path', type=str,
                    default="./pca/pca_proj_mat.npy")
parser.add_argument('--use_whitening', type=bool, default=False)
parser.add_argument('--pca_variances_path', type=str,
                    default="./pca/pca_proj_mat.npy")

args = parser.parse_known_args()[0]


def ReadFromFile(file_path):
    """ReadFromFile"""
    data = np.load(file_path)
    return data


def CalculateReceptiveBoxes(height, width, rf, stride, padding):
    """CalculateReceptiveBoxes"""
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coordinates = np.reshape(np.stack([y, x], axis=2), [-1, 2])
    point_boxes = np.concatenate(
        (coordinates, coordinates), 1).astype(np.float32)
    bias = np.array([-padding, -padding, -padding + rf -
                     1, -padding + rf - 1]).astype(np.float32)
    rf_boxes = stride * point_boxes + bias
    return rf_boxes


def ReadImageList(list_path):
    """ReadImageList"""
    f = open(list_path, "r")
    image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths


def ReadConfig():
    """ReadConfig"""
    config = delf_config.config()
    local_pca_parameters = {}
    if args.use_pca:
        local_pca_parameters['mean'] = np.array(
            ReadFromFile(args.mean_path), np.float32)
        local_pca_parameters['matrix'] = np.array(
            ReadFromFile(args.projection_matrix_path), np.float32)
        local_pca_parameters['dim'] = 40
        local_pca_parameters['use_whitening'] = args.use_whitening
        if args.use_whitening:
            local_pca_parameters['variances'] = np.squeeze(
                np.array(
                    ReadFromFile(args.pca_variances_path),
                    np.float32)
            )
        else:
            local_pca_parameters['variances'] = None
    image_scales_tensor = np.array(config.image_scales, np.float32)
    score_threshold_tensor = config.score_threshold
    max_feature_num = config.max_feature_num
    return (local_pca_parameters, image_scales_tensor,
            score_threshold_tensor, max_feature_num)


def CalculateKeypoint(image_scales_tensor, size_list, attention_prob_batch,
                      feature_map_batch, score_threshold_tensor):
    """CalculateKeypoint"""
    stride_factor = 2.0
    DIM_FEATURE = 1024
    rf, stride, padding = [291.0, 16.0 * stride_factor, 145.0]
    feature_depth = DIM_FEATURE

    for i in range(image_scales_tensor.shape[0]):
        scale = image_scales_tensor[i]
        scale_size = size_list[i]
        attention_prob = attention_prob_batch[i, :, :, :]
        #print('attention_prob.shape: ', attention_prob.shape)
        feature_map = feature_map_batch[i, :, :, :]

        input_perm = (1, 2, 0)
        attention_prob = np.transpose(attention_prob, input_perm)
        feature_map = np.transpose(feature_map, input_perm)

        attention_prob = attention_prob[:(scale_size[0]*attention_prob.shape[0]//2048),
                                        :(scale_size[1]*attention_prob.shape[1]//2048),]
        feature_map = feature_map[:(scale_size[0]*feature_map.shape[0]//2048),
                                  :(scale_size[1]*feature_map.shape[1]//2048),]

        rf_boxes_glob = CalculateReceptiveBoxes(
            feature_map.shape[0],
            feature_map.shape[1], rf, stride, padding)

        # Re-project back to the original image space.
        rf_boxes = rf_boxes_glob / scale
        attention_prob = np.reshape(attention_prob, (-1,))
        feature_map = np.reshape(feature_map, (-1, feature_depth))

        # Use attention score to select feature vectors.
        indices = np.reshape(
            np.where(attention_prob >= score_threshold_tensor), (-1,))
        new_indices = indices

        selected_boxes = rf_boxes[new_indices].astype(np.float32)
        selected_features = feature_map[new_indices].astype(np.float32)
        selected_scores = attention_prob[new_indices].astype(np.float32)
        selected_scales = (np.ones_like(
            selected_scores).astype(np.float32) / scale).astype(np.float32)

        if i == 0:
            output_boxes = selected_boxes
            output_features = selected_features
            output_scales = selected_scales
            output_scores = selected_scores
        else:
            # Concat with the previous result from different scales.
            output_boxes = np.concatenate((output_boxes, selected_boxes), 0)
            output_features = np.concatenate(
                (output_features, selected_features), 0)
            output_scales = np.concatenate((output_scales, selected_scales), 0)
            output_scores = np.concatenate((output_scores, selected_scores), 0)

    return output_boxes, output_features, output_scales, output_scores


def main():
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)

    print('Postprocess: Reading list of images...')
    if args.use_list_txt == "True":
        image_paths = ReadImageList(args.list_images_path)
    else:
        names = os.listdir(args.images_path)
        image_paths = []
        for name in names:
            image_name = name.replace('.jpg', '')
            image_paths.append(image_name)

    (local_pca_parameters, image_scales_tensor,
     score_threshold_tensor, max_feature_num) = ReadConfig()

    num_images = len(image_paths)

    start = time.time()

    for i in range(num_images):
        if i % _STATUS_CHECK_ITERATIONS == 0:
            elapsed = (time.time() - start)
            print(f'Processing image {i} out of {num_images}, last '
                  f'{_STATUS_CHECK_ITERATIONS} images took {elapsed} seconds')
            start = time.time()

        attention_prob_batch = None
        feature_map_batch = None
        raw_local_descriptors = None
        descriptors_out = None
        output_boxes = None
        output_features = None
        output_scales = None
        output_scores = None
        feature_boxes = None
        while (not os.path.exists(
                os.path.join(args.bin_path, image_paths[i])+'_attention.bin') or
               not os.path.exists(
                   os.path.join(args.bin_path, image_paths[i])+'_feature.bin')):
            time.sleep(5)

        attention_prob_batch = np.fromfile(os.path.join(
            args.bin_path, image_paths[i])+'_attention.bin', dtype=np.float32)
        attention_prob_batch.shape = 7, 1, 64, 64
        feature_map_batch = np.fromfile(os.path.join(
            args.bin_path, image_paths[i])+'_feature.bin', dtype=np.float32)
        feature_map_batch.shape = 7, 1024, 64, 64

        os.remove(os.path.join(
            args.bin_path, image_paths[i])+'_attention.bin')
        os.remove(os.path.join(
            args.bin_path, image_paths[i])+'_feature.bin')

        print("post_process: ", image_paths[i])

        # If descriptor already exists, skip its computation.
        out_desc_filename = os.path.splitext(
            os.path.basename(image_paths[i]))[0]
        out_desc_fullpath = os.path.join(args.target_path, out_desc_filename)
        if os.path.exists(out_desc_fullpath+'.feature'+'.npz'):
            print(f'Skipping {image_paths[i]}')
            continue

        size_list = np.load(os.path.join(
            args.size_path, image_paths[i])+'.npy')

        (output_boxes, output_features, output_scales, output_scores) = CalculateKeypoint(
            image_scales_tensor, size_list, attention_prob_batch, feature_map_batch, score_threshold_tensor)

        feature_boxes = box_list.BoxList(output_boxes)
        feature_boxes.add_field('features', output_features)
        feature_boxes.add_field('scales', output_scales)
        feature_boxes.add_field('scores', output_scores)

        iou = 1.0
        nms_max_boxes = min(max_feature_num, feature_boxes.num_boxes())
        final_boxes = box_list_ops.non_max_suppression(
            feature_boxes, iou, nms_max_boxes)
        boxes = final_boxes.get()
        raw_local_descriptors = final_boxes.get_field('features')
        feature_scales_out = final_boxes.get_field('scales')
        attention_with_extra_dim = final_boxes.get_field(
            'scores').reshape((-1, 1))

        attention_out = np.reshape(
            attention_with_extra_dim, (attention_with_extra_dim.shape[0],))
        locations_out, descriptors_out = (
            extract_utils.DelfFeaturePostProcessing(
                boxes, raw_local_descriptors, args.use_pca, local_pca_parameters))

        feature_io.WriteToFile(out_desc_fullpath, locations_out, feature_scales_out,
                               descriptors_out, attention_out)


if __name__ == "__main__":
    main()
