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
''' trans operation '''
import sys
import math
import random
import collections
import os
import os.path as osp
import cv2
import numpy as np

if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable


def read_image_color(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                img_path))
    return img


def read_image_grayscale(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                img_path))
    return img


def is_convex(kpts):
    ''' is convex '''
    dx0 = kpts[-1][0] - kpts[-2][0]
    dy0 = kpts[-1][1] - kpts[-2][1]
    dx1 = kpts[0][0] - kpts[-1][0]
    dy1 = kpts[0][1] - kpts[-1][1]
    x_prod = dx0 * dy1 - dy0 * dx1
    sign_init = np.sign(x_prod)
    if sign_init == 0:
        return False

    for k in range(1, len(kpts)):
        dx0 = kpts[k-1][0] - kpts[k-2][0]
        dy0 = kpts[k-1][1] - kpts[k-2][1]
        dx1 = kpts[k][0] - kpts[k-1][0]
        dy1 = kpts[k][1] - kpts[k-1][1]
        x_prod = dx0 * dy1 - dy0 * dx1
        if sign_init != np.sign(x_prod):
            return False

    return True


def to_tensor(img, output_type=np.float32):
    ''' hwc2chw and norm '''
    return (img.transpose(2, 0, 1).copy() / 255.).astype(output_type)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean_pe = np.array(mean, dtype=np.float32)
std_pe = np.array(std, dtype=np.float32)
mean_avg = sum(mean) / float(3)
std_avg = sum(std) / float(3)
mean.extend([mean_avg] * 49)
std.extend([std_avg] * 49)
mean_mt = np.array(mean, dtype=np.float32)
std_mt = np.array(std, dtype=np.float32)


def normalize_pe_input(img):
    ''' norm '''
    return (img - mean_pe[:, None, None]) / std_pe[:, None, None]


def normalize_mt_input(img, heatmapaware=True, segmentaware=True):
    ''' norm '''
    if heatmapaware and segmentaware:
        image = (img - mean_mt[:, None, None]) / std_mt[:, None, None]
    elif heatmapaware:
        image = (img - mean_mt[:39, None, None]) / std_mt[:39, None, None]
    else:
        image = (img - mean_mt[:16, None, None]) / std_mt[:16, None, None]
    # if padding_channel: padding_chnl == false
    #     zeros = np.zeros([1, image.shape[1], image.shape[2]], dtype=np.float32)
    #     image = np.concatenate((image, zeros), axis=0)
    return image


def image_proc(img_path, test=True):
    ''' get PoseEstNet input data '''
    data_numpy = cv2.imread(
        img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if test:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    if data_numpy is None:
        raise ValueError('Fail to read {}'.format(img_path))
    height, width, _ = data_numpy.shape

    center = np.zeros((2), dtype=np.float32)
    center[0] = width * 0.5
    center[1] = height * 0.5
    pixel_std = 200

    if width > height:
        height = width * 1.0
    elif width < height:
        width = height * 1.0
    scale = np.array(
        [width * 1.0 / pixel_std, height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    r = 0
    trans = get_affine_transform(center, scale, r, [256, 256], inv=0)
    orig_img_hwc = data_numpy
    img_hwc = cv2.warpAffine(
        data_numpy,
        trans,
        (256, 256),
        flags=cv2.INTER_LINEAR)
    img_chw = to_tensor(img_hwc)
    pn_input = normalize_pe_input(img_chw)

    return orig_img_hwc, pn_input, center, scale


flip_pairs = [[0, 18], [1, 19], [2, 20], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25], [8, 26], [9, 27],
              [10, 28], [11, 29], [12, 30], [13, 31], [14, 32], [15, 33], [16, 34], [17, 35]]


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    ''' trans coors in 256*256 to orig_height*orig_width '''
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    ''' get affine transform '''
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        # print(scale)
        scale = np.array([scale, scale])
    #  height of the original image
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    ''' affine transform '''
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    ''' get 3rd point of affine transform '''
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    ''' get dir '''
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = np.array([0, 0], np.float32)
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_posenet_preds(batch_heatmaps, center=None, scale=None, test_postprocess=True, trans_back=True):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    # batchsize*numjoints*1
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))
    # batchsize*numjoints*2
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    if not trans_back:
        return preds, maxvals
    heatmap_width = batch_heatmaps.shape[2]
    heatmap_height = batch_heatmaps.shape[3]
    if test_postprocess:
        for n in range(preds.shape[0]):
            for p in range(preds.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(preds[n][p][0] + 0.5))

                py = int(math.floor(preds[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    preds[n][p] += np.sign(diff) * .25

    for i in range(preds.shape[0]):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [64, 64]
        )
    all_preds = np.zeros(
        (preds.shape[0], 36, 3),
        dtype=np.float32
    )
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    return all_preds


segs = [(5, 15, 16, 17), (5, 6, 12, 15), (6, 10, 11, 12),
        (23, 33, 34, 35), (23, 24, 30, 33), (24, 28, 29, 30),
        (10, 11, 29, 28), (11, 12, 30, 29), (12, 13, 31, 30),
        (13, 14, 32, 31), (14, 15, 33, 32), (15, 16, 34, 33),
        (16, 17, 35, 34)]


def gene_mt_input(imgs, heatmaps, vkeypts, heatmapaware=True, segmrntaware=True):
    ''' get MultiTaskNet input data '''
    batch, height, width, _ = imgs.shape
    img_seg_size = (64, 64)
    mt_inputs = []
    vkeypts = vkeypts.reshape([batch, -1])
    if heatmapaware:
        heatmaps = heatmaps * 255
        heatmaps = np.clip(heatmaps, a_min=0, a_max=255)
        heatmaps = heatmaps.astype(np.uint8)
    for b in range(batch):
        img_chnls = []
        img_r, img_g, img_b = cv2.split(imgs[b])
        img_chnls.extend([img_r, img_g, img_b])

        if heatmapaware:
            for i in range(36):
                heatmap = cv2.resize(heatmaps[b][i], dsize=(width, height))
                img_chnls.append(heatmap)

        if segmrntaware:
            kpts = []
            for k in range(36):
                kpt = (int(round(float(vkeypts[b][k*3]))),
                       int(round(float(vkeypts[b][k*3+1]))))
                kpts.append(kpt)

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

                segment_flag = True
                for k in segs[s]:
                    # threld=0.5
                    if vkeypts[b][k * 3 + 2] < 0.5:
                        segment_flag = False
                        break
                if segment_flag:
                    segment = cv2.resize(img_seg, dsize=(width, height))
                else:
                    segment = np.zeros((height, width), np.uint8)
                img_chnls.append(segment)

            # assert self.transform is not None
        img = np.stack(img_chnls, axis=2)
        img = np.array(img, np.float32)
        img = (Resize_Keypt((256, 256)))(img, vkeypts[b])
        img = to_tensor(img)
        img = normalize_mt_input(img, heatmapaware, segmrntaware)
        # normalize keypt
        for k in range(vkeypts[b].size):
            if k % 3 == 0:
                vkeypts[b][k] = (vkeypts[b][k] / float(256)) - 0.5
            elif k % 3 == 1:
                vkeypts[b][k] = (vkeypts[b][k] / float(256)) - 0.5
            elif k % 3 == 2:
                vkeypts[b][k] -= 0.5
        mt_inputs.append(img)
    return mt_inputs, vkeypts


class Compose_Keypt():
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, vkeypt):
        for t in self.transforms:
            img = t(img, vkeypt)
        return img


class Resize_Keypt():
    """
    Resize the input image to the given size.

    Args:
        size (sequence): Desired output size (w, h).
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, Iterable) and len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, vkeypt):
        """
        Args:
            img (NumPy array): Image to be scaled.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if not len(vkeypt) == 108:
            raise TypeError(
                'vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

        height, width, _ = img.shape

        width_scale = float(self.size[0]) / float(width)
        height_scale = float(self.size[1]) / float(height)
        for k in range(len(vkeypt)):
            if k % 3 == 0:
                vkeypt[k] *= width_scale
            elif k % 3 == 1:
                vkeypt[k] *= height_scale
        return cv2.resize(img, dsize=self.size, interpolation=self.interpolation)


class RandomHorizontalFlip_Keypt():
    """
    Horizontally flip the given image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, vkeypt):
        """
        Args:
            img (NumPy array): Image to be flipped.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if random.random() < self.p:
            if not len(vkeypt) == 108:
                raise TypeError(
                    'vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

            _, width, _ = img.shape
            for k in range(len(vkeypt)):
                if k % 3 == 0:
                    vkeypt[k] = width - vkeypt[k] - 1

            img = cv2.flip(img, flipCode=1)

        return img


class Random2DTranslation_Keypt():
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        size (sequence): Desired output size (w, h).
        p (float): probability of performing this transformation. Default: 0.5.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, p=0.5, interpolation=cv2.INTER_LINEAR):
        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img, vkeypt):
        """
        Args:
            img (NumPy array): Image to be cropped.
            vkeypot (36x3 list): 2D keypioints and confidence scores.
        """
        if not len(vkeypt) == 108:
            raise TypeError(
                'vkeypt should have size 108. Got inappropriate size: {}'.format(len(vkeypt)))

        height, width, _ = img.shape
        # print(img.shape)

        if random.uniform(0, 1) > self.p:
            width_scale = float(self.size[0]) / float(width)
            height_scale = float(self.size[1]) / float(height)
            for k in range(len(vkeypt)):
                if k % 3 == 0:
                    vkeypt[k] *= width_scale
                elif k % 3 == 1:
                    vkeypt[k] *= height_scale

            return cv2.resize(img, dsize=self.size, interpolation=self.interpolation)

        width_new, height_new = int(
            round(self.size[0] * 1.125)), int(round(self.size[1] * 1.125))

        width_scale = float(width_new) / float(width)
        height_scale = float(height_new) / float(height)
        for k in range(len(vkeypt)):
            if k % 3 == 0:
                vkeypt[k] *= width_scale
            elif k % 3 == 1:
                vkeypt[k] *= height_scale

        img_resized = cv2.resize(img, dsize=(
            width_new, height_new), interpolation=self.interpolation)
        x_maxrange = width_new - self.size[0]
        y_maxrange = height_new - self.size[1]
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))

        for k in range(len(vkeypt)):
            if k % 3 == 0:
                vkeypt[k] -= x1
            elif k % 3 == 1:
                vkeypt[k] -= y1

        return img_resized[y1: (y1 + self.size[1]), x1: (x1 + self.size[0])]


def save_heatmaps(batch_heatmaps, batch_image_names, output_heapmap_dir):
    '''save heatmaps'''
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]

    for i in range(batch_size):
        heatmaps = batch_heatmaps[i]

        heatmaps = heatmaps * 255
        heatmaps = np.clip(heatmaps, a_min=0, a_max=255)
        heatmaps = heatmaps.astype(np.uint8)
        output_sub_dir = os.path.join(
            output_heapmap_dir, batch_image_names[i][:-4])

        if not os.path.exists(output_sub_dir):
            os.mkdir(output_sub_dir)

        for j in range(num_joints):
            file_name = os.path.join(output_sub_dir,
                                     "%02d.jpg" % j)
            cv2.imwrite(file_name, heatmaps[j, :, :])
