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
"""extract features from images using delf"""
import argparse
import time
import os
from  multiprocessing import Process, Queue

import extract_utils_np as extract_utils
import box_list_np as box_list
import box_list_ops_np as box_list_ops
import feature_io
import delf_config
import delf_model as delf

from PIL import Image
import numpy as np

from mindspore import load_checkpoint, load_param_into_net
from mindspore import context, Tensor

_STATUS_CHECK_ITERATIONS = 10

parser = argparse.ArgumentParser(description='MindSpore delf Example')

parser.add_argument('--devices', type=str, default="0")

parser.add_argument('--use_list_txt', type=str, default="True", choices=['True', 'False'])
parser.add_argument('--list_images_path', type=str, default="list_images.txt")

parser.add_argument('--ckpt_path', type=str, default="")
parser.add_argument('--images_path', type=str, default="")
parser.add_argument('--target_path', type=str, default="")

# pca config
parser.add_argument('--use_pca', type=bool, default=True)
parser.add_argument('--mean_path', type=str, default="./pca/mean.npy")
parser.add_argument('--projection_matrix_path', type=str, default="./pca/pca_proj_mat.npy")
parser.add_argument('--use_whitening', type=bool, default=False)
parser.add_argument('--pca_variances_path', type=str, default="./pca/pca_proj_mat.npy")

args = parser.parse_known_args()[0]


def ReadImageList(list_path):
    f = open(list_path, "r")
    image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths

def ReadFromFile(file_path):
    data = np.load(file_path)
    return data

def CalculateReceptiveBoxes(height, width, rf, stride, padding):
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coordinates = np.reshape(np.stack([y, x], axis=2), [-1, 2])
    point_boxes = np.concatenate((coordinates, coordinates), 1).astype(np.float32)
    bias = np.array([-padding, -padding, -padding + rf - 1, -padding + rf - 1]).astype(np.float32)
    rf_boxes = stride * point_boxes + bias
    return rf_boxes

def preprocess(q, num_images, image_paths, image_scales_tensor):
    """preprocess images"""
    for i in range(num_images):
        # Report progress once in a while.
        if i == 0:
            print('Starting to extract DELF features from images...')

        # If descriptor already exists, skip its computation.
        out_desc_filename = os.path.splitext(os.path.basename(
            image_paths[i]))[0]
        out_desc_fullpath = os.path.join(args.target_path, out_desc_filename)
        if os.path.exists(out_desc_fullpath+'.feature'+'.npz'):
            print(f'Skipping {image_paths[i]}')
            continue

        img = Image.open(os.path.join(args.images_path, image_paths[i]) + '.jpg')

        im = np.array(img, np.float32)
        original_image_shape = np.array([im.shape[0], im.shape[1]])
        original_image_shape_float = original_image_shape.astype(np.float32)
        new_image_size = np.array([2048, 2048])

        images_batch = np.zeros((image_scales_tensor.shape[0], 3, 2048, 2048), np.float32)
        size_list = []

        for j in range(image_scales_tensor.shape[0]):
            scale_size = np.round(original_image_shape_float * image_scales_tensor[j]).astype(int)
            size_list.append(scale_size)
            img_pil = img.resize((scale_size[1], scale_size[0]))
            scale_image = np.array(img_pil, np.float32)

            H_pad = new_image_size[0] - scale_size[0]
            W_pad = new_image_size[1] - scale_size[1]
            new_image = np.pad(scale_image, ((0, H_pad), (0, W_pad), (0, 0)))

            new_image = (new_image-128.0) / 128.0

            perm = (2, 0, 1)
            new_image = np.transpose(new_image, perm)
            new_image = np.expand_dims(new_image, 0)
            images_batch[j] = new_image
        print('preprocess: images_batch: ', images_batch.shape)
        img_dict = {'image': images_batch, 'path': out_desc_fullpath, 'size_list': size_list}
        q.put(img_dict)

def model_predict(q, q1, device):
    """model predict"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device)

    delf_model = delf.Model('test')
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(delf_model, param_dict)
    print('model_predict: init suceesully! ')
    while True:
        food = q.get()
        if food is None:
            break
        images_batch = food['image']
        attention_prob_batch, feature_map_batch = delf_model(Tensor(images_batch))
        attention_prob_batch = attention_prob_batch.asnumpy()
        feature_map_batch = feature_map_batch.asnumpy()
        img_dict = {'attention': attention_prob_batch, 'feature': feature_map_batch,
                    'path': food['path'], 'size_list': food['size_list']}
        print('model_predict: done')
        q1.put(img_dict)

def postprocess(q, num_images, image_scales_tensor,
                score_threshold_tensor, max_feature_num, local_pca_parameters):
    """postprocess images"""
    iou = 1.0
    stride_factor = 2.0
    DIM_FEATURE = 1024
    rf, stride, padding = [291.0, 16.0 * stride_factor, 145.0]
    feature_depth = DIM_FEATURE

    start = time.time()
    i_t = 0
    while True:
        food = q.get()
        if food is None:
            break
        i_t += 1
        if i_t % _STATUS_CHECK_ITERATIONS == 0:
            elapsed = (time.time() - start)
            print(f'Processing image {i_t} out of {num_images}, last '
                  f'{_STATUS_CHECK_ITERATIONS} images took {elapsed} seconds')
            start = time.time()

        attention_prob_batch = food['attention']
        feature_map_batch = food['feature']
        out_desc_fullpath = food['path']
        size_list = food['size_list']

        # calculate keypoints
        for i in range(image_scales_tensor.shape[0]):
            scale = image_scales_tensor[i]
            scale_size = size_list[i]
            attention_prob = attention_prob_batch[i, :, :, :]
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
            indices = np.reshape(np.where(attention_prob >= score_threshold_tensor), (-1,))
            new_indices = indices

            selected_boxes = rf_boxes[new_indices].astype(np.float32)
            selected_features = feature_map[new_indices].astype(np.float32)
            selected_scores = attention_prob[new_indices].astype(np.float32)
            selected_scales = (np.ones_like(selected_scores).astype(np.float32) / scale) .astype(np.float32)

            if i == 0:
                output_boxes = selected_boxes
                output_features = selected_features
                output_scales = selected_scales
                output_scores = selected_scores
            else:
                # Concat with the previous result from different scales.
                output_boxes = np.concatenate((output_boxes, selected_boxes), 0)
                output_features = np.concatenate((output_features, selected_features), 0)
                output_scales = np.concatenate((output_scales, selected_scales), 0)
                output_scores = np.concatenate((output_scores, selected_scores), 0)

        # nms
        feature_boxes = box_list.BoxList(output_boxes)
        feature_boxes.add_field('features', output_features)
        feature_boxes.add_field('scales', output_scales)
        feature_boxes.add_field('scores', output_scores)

        nms_max_boxes = min(max_feature_num, feature_boxes.num_boxes())
        final_boxes = box_list_ops.non_max_suppression(feature_boxes, iou,
                                                       nms_max_boxes)
        boxes = final_boxes.get()
        raw_local_descriptors = final_boxes.get_field('features')
        feature_scales_out = final_boxes.get_field('scales')
        attention_with_extra_dim = final_boxes.get_field('scores').reshape((-1, 1))

        attention_out = np.reshape(attention_with_extra_dim,
                                   (attention_with_extra_dim.shape[0],))
        locations_out, descriptors_out = (
            extract_utils.DelfFeaturePostProcessing(
                boxes, raw_local_descriptors, args.use_pca,
                local_pca_parameters))

        feature_io.WriteToFile(out_desc_fullpath, locations_out, feature_scales_out,
                               descriptors_out, attention_out)
        print('postprocess: ', i_t, ' done')

def main():
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)

    # Read list of images.
    print('Preprocess: Reading list of images...')
    if args.use_list_txt == "True":
        image_paths = ReadImageList(args.list_images_path)
    else:
        names = os.listdir(args.images_path)
        image_paths = []
        for name in names:
            if name.endswith('.jpg'):
                image_name = name.replace('.jpg', '')
                image_paths.append(image_name)

    num_images = len(image_paths)
    print(f'done! Found {num_images} images')

    # extract config
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

    # use producer-consumer to extract feature
    q = Queue(maxsize=20)
    q1 = Queue(maxsize=100)

    # producer produce preprocessed images
    p = Process(target=preprocess, args=(q, num_images,
                                         image_paths, image_scales_tensor))
    p.start()

   # use multi-npu to predict
    for i in list(args.devices):
        c = Process(target=model_predict, args=(q, q1, int(i)))
        c.start()
    c1 = Process(target=postprocess, args=(
        q1, num_images, image_scales_tensor, score_threshold_tensor, max_feature_num, local_pca_parameters))

    c1.start()

    p.join()
    for _ in list(args.devices):
        q.put(None)
    time.sleep(5)
    q1.put(None)


if __name__ == "__main__":
    main()
