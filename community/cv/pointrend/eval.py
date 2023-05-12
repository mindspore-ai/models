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
"""Evaluation for MaskRcnn"""

import time
import os
import numpy as np
import mindspore
from mindspore import context, Tensor, ops, numpy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from maskrcnn.model_utils.config import config
from maskrcnn.model_utils.moxing_adapter import moxing_wrapper
from maskrcnn.model_utils.device_adapter import get_device_id, get_device_num
from maskrcnn_pointrend.src.maskrcnnPointRend_r50 import maskrcnn_r50_pointrend
from maskrcnn_pointrend.src.dataset import data_to_mindrecord_byte_image, create_maskrcnn_dataset
from maskrcnn_pointrend.src.util import coco_eval, output2json
from maskrcnn_pointrend.src.point_rend.sampling_points import GridSampler
set_seed(1)
BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024 ** 3
dataset_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
              39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
              57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
              78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def _postprocess(instances, batched_inputs, img_metas):
    """
    Rescale the output instances to the target size.
    """
    processed_results = []

    num_img = batched_inputs.shape[0]
    for i in range(num_img):
        pred_boxes, pred_classes, pred_scores, pred_masks = instances
        results_per_image = (
            pred_boxes[i].asnumpy(), pred_classes[i].asnumpy(),
            pred_scores[i].asnumpy(), pred_masks[i].asnumpy())
        r = detector_postprocess(results_per_image, img_metas[i])
        if r:
            processed_results.append(r)
    return processed_results


def detector_postprocess(results, img_metas):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    output_height_tmp = img_metas[0]
    output_width_tmp = img_metas[1]
    new_size = (output_height_tmp, output_width_tmp)
    output_boxes = results[0]
    scale_h = img_metas[2]
    scale_w = img_metas[3]
    output_boxes[:, 0::2] /= scale_w
    output_boxes[:, 1::2] /= scale_h
    output_boxes = clip(output_boxes, img_metas[:-2])
    keep = nonempty(output_boxes)
    if not keep.any():
        print("no useful data")
        return []
    temp = []
    for item in results:
        item = item[keep]
        temp.append(item)
    results = temp
    pred_boxes = results[0]
    pred_masks = results[3]
    pred_masks = paste_masks_in_image(pred_masks[:, 0, :, :], pred_boxes, new_size)
    results[3] = pred_masks
    return results

def clip(output_boxes, box_size) -> None:
    """
    Clip (in place) the boxes by limiting x coordinates to the range [0, width]
    and y coordinates to the range [0, height].

    Args:
        box_size (height, width): The clipping box's size.
    """
    output_boxes = Tensor(output_boxes)
    h, w = box_size
    x1 = output_boxes[:, 0].clip(xmin=0, xmax=w)
    y1 = output_boxes[:, 1].clip(xmin=0, xmax=h)
    x2 = output_boxes[:, 2].clip(xmin=0, xmax=w)
    y2 = output_boxes[:, 3].clip(xmin=0, xmax=h)
    stack = ops.Stack(-1)
    output_boxes = stack((x1, y1, x2, y2))
    return output_boxes.asnumpy()

def nonempty(output_boxes, threshold=0.0):
    """
    Find boxes that are non-empty.
    A box is considered empty, if either of its side is no larger than threshold.

    Returns:
        Tensor:
            a binary vector which represents whether each box is empty
            (False) or non-empty (True).
    """
    widths = output_boxes[:, 2] - output_boxes[:, 0]
    heights = output_boxes[:, 3] - output_boxes[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep

def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape
    num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
    assert (
        num_chunks <= N
    ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    split = ops.Split(0, num_chunks)
    chunks = split(numpy.arange(N))

    img_masks = numpy.zeros((N, int(img_h), int(img_w)), mindspore.bool_).asnumpy()
    for inds in chunks:
        inds = inds.asnumpy()
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], int(img_h), int(img_w)
        )

        if threshold >= 0:
            masks_chunk = masks_chunk >= threshold
        else:
            masks_chunk = (masks_chunk * 255)
        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks

def _do_paste_mask(masks, boxes, img_h: int, img_w: int):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """

    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    split = ops.Split(1, 4)
    x0, y0, x1, y1 = split(Tensor(boxes))

    N = masks.shape[0]

    img_y = numpy.arange(y0_int, y1_int, dtype=mindspore.float32) + 0.5
    img_x = numpy.arange(x0_int, x1_int, dtype=mindspore.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    broadcast_to = ops.BroadcastTo((N, img_y.shape[1], img_x.shape[1]))
    stack = ops.Stack(3)
    gx = broadcast_to(img_x[:, None, :])
    gy = broadcast_to(img_y[:, :, None])
    grid = stack([gx, gy])
    gridSample = GridSampler(align_corners=False)
    img_masks = gridSample(Tensor(masks), grid)
    img_masks = img_masks[:, 0]
    logic_and = ops.LogicalAnd()
    mask_x = logic_and(grid[..., 0] >= -1, grid[..., 0] <= 1)
    mask_y = logic_and(grid[..., 1] >= -1, grid[..., 1] <= 1)
    mask_xy = logic_and(mask_x, mask_y).astype(mindspore.float32)
    img_masks = img_masks * mask_xy
    return img_masks.asnumpy(), ()

def maskrcnn_eval(dataset_path, ckpt_path, ann_file):
    """MaskRcnn evaluation."""
    print('\nconfig:\n', config)
    ds = create_maskrcnn_dataset(dataset_path, batch_size=config.test_batch_size, is_training=False)
    net = maskrcnn_r50_pointrend(config)
    for item1 in net.parameters_and_names():
        print(item1)
        break
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    for item in net.parameters_and_names():
        print(item)
        break
    net.set_train(False)

    eval_iter = 0
    total = ds.get_dataset_size()
    outputs = []
    dataset_coco = COCO(ann_file)

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")

    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):

        img_id = data['image_id'][0].tolist()
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']
        gt_mask = data["mask"]

        start = time.time()
        output = net(Tensor(img_data), Tensor(img_metas), Tensor(gt_bboxes), Tensor(gt_labels), Tensor(gt_num),
                     Tensor(gt_mask))
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start))
        del gt_bboxes
        del gt_labels
        del gt_mask
        del gt_num
        if not output:
            continue
        processed_results = _postprocess(output, img_data, img_metas)
        del img_data
        if not processed_results:
            continue
        result = process(processed_results, img_id)
        outputs.append(result)
        eval_iter = eval_iter + 1
    if not outputs:
        return
    eval_types = ["bbox", "segm"]
    result_files = output2json(outputs, "./results.pkl", dataset_id)
    coco_eval(result_files, eval_types, dataset_coco, single_result=False)


def process(outputs, img_id):
    """
    Args:
        inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
            It is a list of dict. Each dict corresponds to an image and
            contains keys like "height", "width", "file_name", "image_id".
        outputs: the outputs of a COCO model. It is a list of dicts with key
            "instances" that contains :class:`Instances`.
    """
    predictions = []
    for output in outputs:
        segm_json_results, box_json_results = instances_to_coco_json(output, img_id)
        if segm_json_results:
            predictions.append((segm_json_results, box_json_results))
    return predictions

def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances[0])
    if num_instance == 0:
        return []
    boxes = instances[0]
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    classes = instances[1]
    scores = instances[2]
    pred_masks = instances[3]
    classes = classes.tolist()
    scores = scores.tolist()
    boxes = boxes.tolist()
    rles = [
        maskUtils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        for mask in pred_masks
    ]
    for rle in rles:
        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        # json writer which always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does).
        rle["counts"] = rle["counts"].decode("utf-8")

    segm_json_results = []
    box_json_results = []
    for k in range(num_instance):
        seg_result = {
            "image_id": img_id,
            "category_id": classes[k],
            "score": scores[k],
        }
        box_result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        box_json_results.append(box_result)
        seg_result["segmentation"] = rles[k]
        segm_json_results.append(seg_result)
    return segm_json_results, box_json_results

def modelarts_process():
    """ modelarts process """

    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60), \
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))
        print("#" * 200, os.listdir(save_dir_1))
        print("#" * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

        config.coco_root = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    config.checkpoint_path = os.path.join(config.output_path, config.checkpoint_path)
    config.ann_file = os.path.join(config.coco_root, config.ann_file)


@moxing_wrapper(pre_process=modelarts_process)
def eval_():
    '''eval'''
    device_target = config.device_target
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)

    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)
    else:
        context.set_context(device_id=0)

    prefix = "MaskRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    print("Start Eval!")
    ann_file = config.coco_root + "/annotations/instances_val2017.json"
    maskrcnn_eval(mindrecord_file, config.checkpoint_path, ann_file)

if __name__ == '__main__':
    eval_()
