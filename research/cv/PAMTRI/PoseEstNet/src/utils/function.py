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
"""function"""
import os
import csv
import cv2
import numpy as np

from mindspore import Tensor
from src.dataset import flip_back, get_final_preds
from .vis import is_convex, save_heatmaps

def _print_name_value(name_value, full_arch_name):
    """_print_name_value"""
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('| --- ' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )

def validate(config, val_loader, val_dataset, model, allImage):
    """validate"""
    model.set_train(False)

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    for i, data in enumerate(val_loader.create_dict_iterator()):
        _input = data["input"]
        center = data["center"]
        scale = data["scale"]
        score = data["score"]
        image_label = data["image"]
        joints = data["joints"]
        joints_vis = data["joints_vis"]

        joints = joints.asnumpy()
        joints_vis = joints_vis.asnumpy()

        image = []
        for j in range(config.TEST.BATCH_SIZE):
            image.append(allImage['{}'.format(image_label[j])])

        outputs = model(_input)
        if isinstance(outputs, list):
            print("output is tuple")
            output = outputs[-1]
        else:
            output = outputs

        if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            input_flipped = np.flip(_input.asnumpy(), 3)
            input_flipped = Tensor(input_flipped)
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = flip_back(output_flipped.asnumpy(), val_dataset.flip_pairs)
            output_flipped = Tensor(output_flipped)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP: # true
                output_flipped_copy = output_flipped
                output_flipped[:, :, :, 1:] = output_flipped_copy[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        # measure accuracy and record loss
        num_images = _input.shape[0]

        c = center.asnumpy()
        s = scale.asnumpy()
        score = score.asnumpy()

        output_copy = output
        preds, maxvals = get_final_preds(config, output_copy.asnumpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(image)

        idx += num_images

        print('-------- Test: [{0}/{1}] ---------'.format(i, val_loader.get_dataset_size()))
        name_values, perf_indicator = val_dataset.evaluate(
            all_preds, '', all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

    return perf_indicator

def onnx_validate(config, val_loader, val_dataset, InferenceSession, input_name, is_train, allImage):
    """onnx_validate"""

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    for i, data in enumerate(val_loader.create_dict_iterator()):
        _input = data["input"]
        center = data["center"]
        scale = data["scale"]
        score = data["score"]
        image_label = data["image"]
        joints = data["joints"]
        joints_vis = data["joints_vis"]

        joints = joints.asnumpy()
        joints_vis = joints_vis.asnumpy()

        image = []
        for j in range(config.TEST.BATCH_SIZE):
            image.append(allImage['{}'.format(image_label[j])])

        outputs = InferenceSession.run(None, {input_name: _input.asnumpy()})[0]

        if isinstance(outputs, list):
            print("output is tuple")
            output = outputs[-1]
        else:
            output = outputs
        if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            input_flipped = np.flip(_input.asnumpy(), 3)
            outputs_flipped = InferenceSession.run(None, {input_name: input_flipped})[0]

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = flip_back(output_flipped, val_dataset.flip_pairs)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP: # true
                output_flipped_copy = output_flipped
                output_flipped[:, :, :, 1:] = output_flipped_copy[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        # measure accuracy and record loss
        num_images = _input.shape[0]

        c = center.asnumpy()
        s = scale.asnumpy()
        score = score.asnumpy()

        output_copy = output

        preds, maxvals = get_final_preds(config, output_copy, c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(image)

        idx += num_images

        print('-------- Test: [{0}/{1}] ---------'.format(i, val_loader.get_dataset_size()))
        name_values, perf_indicator = val_dataset.evaluate(
            all_preds, '', all_boxes, image_path,
            filenames, imgnums
        )
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

    return perf_indicator

def output_preds(config, val_loader, val_dataset, model, root, test_set, output_dir):
    """output_preds"""
    gt_file = os.path.join(root, 'label_{}.csv'.format(test_set))

    image_names = []
    with open(gt_file) as annot_file:
        reader = csv.reader(annot_file, delimiter=',')
        for row in reader:
            image_names.append(row[0])

    output_heapmap_dir = os.path.join(output_dir,
                                      'heatmap_{}'.format(test_set))
    if not os.path.exists(output_heapmap_dir):
        os.mkdir(output_heapmap_dir)

    model.set_train(False)

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )

    idx = 0
    for i, data in enumerate(val_loader.create_dict_iterator()):
        _input = data["input"]
        center = data["center"]
        scale = data["scale"]

        outputs = model(_input)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        if config.TEST.FLIP_TEST:
            input_flipped = np.flip(_input.asnumpy(), 3).copy()
            input_flipped = Tensor(input_flipped)
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = flip_back(output_flipped.asnumpy(), val_dataset.flip_pairs)
            output_flipped = Tensor(output_flipped)

            if config.TEST.SHIFT_HEATMAP:
                output_flipped_copy = output_flipped
                output_flipped[:, :, :, 1:] = output_flipped_copy[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        num_images = _input.shape[0]

        c = center.asnumpy()
        s = scale.asnumpy()

        output_copy = output
        preds, maxvals = get_final_preds(config, output_copy.asnumpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals

        batch_image_names = image_names[idx:idx + num_images]
        save_heatmaps(output, batch_image_names, output_heapmap_dir)

        idx += num_images

    output_pose_path = os.path.join(output_dir, 'pose_{}.csv'.format(test_set))

    output_pose = open(output_pose_path, 'w')

    for p in range(len(all_preds)):
        output_pose.write("%s," % (image_names[p]))
        for k in range(len(all_preds[p])-1):
            output_pose.write("%.3f,%.3f,%.3f," % (all_preds[p][k][0],
                                                   all_preds[p][k][1],
                                                   all_preds[p][k][2]))
        output_pose.write("%.3f,%.3f,%.3f\n" % (all_preds[p][len(all_preds[p])-1][0],
                                                all_preds[p][len(all_preds[p])-1][1],
                                                all_preds[p][len(all_preds[p])-1][2]))

    output_pose.close()

    img_seg_size = (64, 64)
    segs = [(5, 15, 16, 17), (5, 6, 12, 15), (6, 10, 11, 12),
            (23, 33, 34, 35), (23, 24, 30, 33), (24, 28, 29, 30),
            (10, 11, 29, 28), (11, 12, 30, 29), (12, 13, 31, 30),
            (13, 14, 32, 31), (14, 15, 33, 32), (15, 16, 34, 33),
            (16, 17, 35, 34)]

    output_segment_dir = os.path.join(output_dir,
                                      'segment_{}'.format(test_set))
    if not os.path.exists(output_segment_dir):
        os.mkdir(output_segment_dir)

    with open(output_pose_path) as input_pose:
        reader = csv.reader(input_pose, delimiter=',')
        for row in reader:
            img_path = os.path.join(root, 'image_' + test_set, row[0])
            img = cv2.imread(img_path)

            height, width, _ = img.shape

            kpts = []
            for k in range(36):
                kpt = (int(round(float(row[k*3+1]))), int(round(float(row[k*3+2]))))
                kpts.append(kpt)

            output_subdir = os.path.join(output_segment_dir, row[0][:-4])
            if not os.path.exists(output_subdir):
                os.mkdir(output_subdir)

            for s in range(len(segs)):
                img_seg = np.zeros([height, width], dtype=np.uint8)
                kpts_seg = []
                for i in segs[s]:
                    kpts_seg.append([kpts[i][0], kpts[i][1]])

                if is_convex(kpts_seg):
                    kpts_seg = np.array([kpts_seg], dtype=np.int32)
                    cv2.fillPoly(img_seg, kpts_seg, 255)
                    img_seg = cv2.resize(img_seg, img_seg_size)
                else:
                    img_seg = np.zeros(img_seg_size, dtype=np.uint8)

                cv2.imwrite(os.path.join(output_subdir, "%02d.jpg" % s), img_seg)
