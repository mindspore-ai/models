# Copyright 2022 Huawei Technologies Co., Ltd
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
"""DGCNet(res101) eval."""
import argparse
import os
import timeit
import json
from math import ceil
import numpy as np
from scipy import ndimage
from PIL import Image as PILImage

import mindspore
from mindspore import Tensor

from mindspore import load_checkpoint, load_param_into_net
from mindspore import context

from src.cityscapes import create_dataset
from src.DualGCNNet import DualSeg_res101


def str2bool(v):
    """str2bool"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        result = True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        result = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    return result


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DGCNet-ResNet101 Network")
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data_list", type=str, default="./src/data/cityscapes/eval.txt",
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data_set", type=str, default="cityscapes", help="dataset to train")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--input_size", type=int, default=832,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--whole", type=bool, default=False,
                        help="use whole input size.")
    parser.add_argument("--rgb", type=int, default=1)

    # ***** Params for save and load ******
    parser.add_argument("--restore_from", type=str, default=None,
                        help="Where restore models parameters from.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="output dir of prediction")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, image, tile_size, classes):
    """predict_sliding"""
    interp = mindspore.ops.ResizeBilinear(tile_size, align_corners=True)
    image_size = image.shape
    overlap = 1.0 / 3.0

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0
    transpose = mindspore.ops.Transpose()

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_img = Tensor(padded_img)
            padded_prediction = list(net(padded_img))
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction)[0]
            padded_prediction = transpose(padded_prediction, (1, 2, 0)).asnumpy()
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def predict_whole(net, image, tile_size):
    """predict_whole"""
    transpose = mindspore.ops.Transpose()
    image = Tensor(image)
    interp = mindspore.ops.ResizeBilinear(tile_size, align_corners=True)
    prediction = net(image)
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy()
    prediction = transpose(prediction, (1, 2, 0))
    return prediction


def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    _, _, H, W = image.shape
    full_probs = np.zeros((H, W, classes))
    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_probs = predict_whole(net, scale_image, tile_size)
        if flip_evaluation:
            flip_scaled_probs = predict_whole(net, scale_image[:, :, :, ::-1].copy(), tile_size)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:, ::-1, :])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the number of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def val():
    """Create the models and start the evaluation process."""
    start = timeit.default_timer()
    args = get_arguments()
    device_id = args.device_id
    target = 'GPU'
    context.set_context(mode=context.GRAPH_MODE, device_target=target, device_id=device_id)

    network = DualSeg_res101(num_classes=args.num_classes, is_train=False)
    saved_state_dict = load_checkpoint(args.restore_from)
    load_param_into_net(network, saved_state_dict)

    # RGB input
    h, w = args.input_size, args.input_size
    input_size = (h, w)
    IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    # set data loader
    test_ds = create_dataset(args, crop_size=(1024, 2048), max_iters=None, mean=IMG_MEAN, vari=IMG_VARS,
                             scale=False, mirror=False)
    testloader = test_ds.create_dict_iterator()
    print("Create test dataset done!")

    # set test net
    test_net = network
    test_net.set_train(False)

    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    palette = get_palette(256)

    output_images = os.path.join(args.output_dir, "./images")
    output_results = os.path.join(args.output_dir, "./result")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(output_images):
        os.makedirs(output_images)
    if not os.path.exists(output_results):
        os.makedirs(output_results)

    for index, data in enumerate(testloader):
        print('%d processd' % (index))
        image = data['image']
        label = data['label'].astype("int64")
        output = predict_sliding(test_net, image.asnumpy(), input_size, args.num_classes)

        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output_im = PILImage.fromarray(seg_pred)
        output_im.putpalette(palette)

        seg_gt = np.asarray(label[0].asnumpy(), dtype=np.int)

        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

    pos = confusion_matrix.sum(axis=1)
    res = confusion_matrix.sum(axis=0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    end = timeit.default_timer()
    print("Eval cost: " + str(end - start) + 'seconds')

    print({'meanIU': mean_IU, 'IU_array': IU_array})
    with open(os.path.join(args.output_dir, "result", 'result.txt'), 'w') as f:
        f.write(json.dumps({'meanIU': mean_IU, 'IU_array': IU_array.tolist()}))


if __name__ == '__main__':
    val()
