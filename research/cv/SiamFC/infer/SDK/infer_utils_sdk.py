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
"""the utils used in dataset"""
import numpy as np
import cv2

class config():
    """set config """
    scale_penalty = 0.9745
    num_scale = 3
    response_up_stride = 16
    response_sz = 17
    exemplar_size = 127
    instance_size = 255
    scale_step = 1.0375
    context_amount = 0.5
    stream_name1 = "init_exemplar"
    stream_name2 = "siamfc_infer"
    exemplar_img_id = 0
    none_id = 1
    exemplar_output_id = 0
    instance_output_id = 1
    TENSOR_DTYPE_FLOAT32 = 0
    device_id = 0
    window_influence = 0.176
    total_stride = 8
    scale_lr = 0.59


def get_center(x):
    """get box center"""
    return (x - 1.) / 2.


def xyxy2cxcywh(bbox):
    """change box format"""
    return get_center(bbox[0]+bbox[2]), \
        get_center(bbox[1]+bbox[3]), \
        (bbox[2]-bbox[0]), \
        (bbox[3]-bbox[1])


def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    """crop and pad image, pad with image mean """
    xmin = cx - original_sz // 2
    xmax = cx + original_sz // 2
    ymin = cy - original_sz // 2
    ymax = cy + original_sz // 2
    im_h, im_w, _ = img.shape

    left = right = top = bottom = 0
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > im_w:
        right = int(xmax - im_w)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > im_h:
        bottom = int(ymax - im_h)

    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = img[ymin:ymax, xmin:xmax]
    if left != 0 or right != 0 or top != 0 or bottom != 0:
        if img_mean is None:

            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=img_mean)
    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    return im_patch


def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    """get exemplar according to given size"""
    cx, cy, w, h = xyxy2cxcywh(bbox)

    wc_z = w + context_amount * (w+h)  # w+0.5*(w+h)
    hc_z = h + context_amount * (w+h)  # h+0.5*(w+h)
    # orginal_sz
    s_z = np.sqrt(wc_z * hc_z)
    # model_sz
    scale_z = size_z / s_z
    exemplar_img = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    return exemplar_img, scale_z, s_z


def get_pyramid_instance_image(img, center, size_x, size_x_scales,
                               img_mean=None):
    """get pyramid instance"""
    if img_mean is None:
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
    pyramid = [crop_and_pad(img, center[0], center[1], size_x,
                            size_x_scale, img_mean)
               for size_x_scale in size_x_scales]
    return pyramid


def get_exemplar(sdk_infer_exemplar, img, none, Config):
    """get exemplar"""
    pipename = Config.stream_name1.encode("utf-8")
    sdk_api = sdk_infer_exemplar
    sdk_api.send_tensor_input(
        pipename, Config.exemplar_img_id, "appsrc0", img.tobytes(), img.shape, Config.TENSOR_DTYPE_FLOAT32)
    sdk_api.send_tensor_input(
        pipename, Config.none_id, "appsrc1", none.tobytes(), none.shape, Config.TENSOR_DTYPE_FLOAT32)
    result = sdk_api.get_result(pipename)
    data = np.frombuffer(
        result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    data = data.reshape((1, 256, 6, 6))
    return data


def do_infer(sdk_infer_instance, exemplar, instance, Config):
    """infer"""
    pipename = Config.stream_name2.encode("utf-8")
    sdk_api = sdk_infer_instance
    sdk_api.send_tensor_input(pipename, Config.exemplar_output_id, "appsrc0", exemplar.tobytes(
    ), exemplar.shape, Config.TENSOR_DTYPE_FLOAT32)
    sdk_api.send_tensor_input(pipename, Config.instance_output_id, "appsrc1", instance.tobytes(
    ), instance.shape, Config.TENSOR_DTYPE_FLOAT32)
    result = sdk_api.get_result(pipename)
    data = np.frombuffer(
        result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    data = data.reshape((1, 17, 17))
    return data


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    """show image define """
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)
    return img
