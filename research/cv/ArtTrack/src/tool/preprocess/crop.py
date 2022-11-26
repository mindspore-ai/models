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

import json
import os
from copy import deepcopy

import numpy as np
from easydict import EasyDict
from imageio import imread, imsave
from PIL import Image

from src.log import log
from src.tool.preprocess.utils import json_default, pad_zeros, util_set_scale


def crop_data(option):
    """
    crop image

    Args:
        option: crop parameters

    """
    log.debug("crop_data()")

    b_train = option["bTrain"]
    ref_height = option["refHeight"]
    delta_crop = option["deltaCrop"]
    b_single = option["bSingle"]
    b_crop_isolated = option["bCropIsolated"]
    b_multi = option["bMulti"]
    b_objpos_offset = option["bObjposOffset"]
    save_dir = option["saveDir"]
    dataset = option["dataset"]
    # annolist = dataset.annolist
    img_list = np.array(dataset['img_list'])

    log.debug('bTrain: %s', b_train)
    log.debug('refHeight: %s', ref_height)
    log.debug('deltaCrop: %s', delta_crop)
    log.debug('bSingle: %s', b_single)
    log.debug('bCropIsolated: %s', b_crop_isolated)
    log.debug('bMulti: %s', b_multi)
    log.debug('bObjposOffset: %s', b_objpos_offset)

    rect_ids = list()
    if b_single:
        mode = 'singlePerson'
        for _, v in enumerate(dataset['img_list']):
            single_person = v['single_person']
            rect_ids.append(single_person)
    else:
        mode = 'multPerson'
        for _, v in enumerate(dataset['img_list']):
            single_person = v['single_person']
            if not v['rect']:
                rect_ids.append([])
            else:
                r_len = len(v['rect'])
                rect_ids.append(np.setdiff1d(
                    np.arange(0, r_len, dtype=np.uint8), single_person))
    rect_ids = np.asarray(rect_ids, dtype=object)
    annalists_full_name = os.path.join(
        save_dir, "annolist-{}-h{}.json".format(mode, ref_height))
    if os.path.exists(annalists_full_name):
        with open(annalists_full_name, 'r') as f:
            try:
                img_list = json.load(f)
                return img_list
            except json.decoder.JSONDecodeError:
                pass

    img_ids1 = filter(lambda item: len(item[1]) != 0, enumerate(rect_ids))
    img_ids1 = map(lambda item: item[0], img_ids1)
    img_ids1 = [i for i in img_ids1]

    img_ids2 = filter(lambda item: item[1]['img_train'] == b_train,
                      enumerate(dataset['img_list']))
    img_ids2 = map(lambda item: item[0], img_ids2)
    img_ids2 = [i for i in img_ids2]

    log.debug("imgidxs1 len: %s imgidxs2 len: %s", len(img_ids1), len(img_ids2))
    imgidxs = np.intersect1d(img_ids1, img_ids2)
    imgidxs_sub = imgidxs

    img_list_subset = util_set_scale(img_list[imgidxs_sub], 200)
    log.debug("img_list_subset shape: %s", img_list_subset.shape)

    if b_train == 0:
        assert False
        result_list = func_crop_data_test()
    else:
        if b_multi == 1:
            func_crop_data_test()
        else:
            result_list = func_crop_data_train(
                option, img_list_subset, imgidxs_sub, rect_ids[imgidxs_sub])

    with open(annalists_full_name, 'w') as f:
        f.write(json.dumps(result_list, default=json_default))
    return result_list


def func_crop_data_test():
    return []


def get_points_all(points):
    points_all = np.array([])
    for f_p in points:
        pp = [f_p['x'], f_p['y']]
        points_all = np.r_[points_all, pp]
    return points_all


def get_rect_position(points):
    points_all = get_points_all(points)
    points_all = points_all.reshape((-1, 2))
    log.debug("points_all: %s", points_all)
    min_x = np.min(points_all[:, 0])
    max_x = np.max(points_all[:, 0])
    min_y = np.min(points_all[:, 1])
    max_y = np.max(points_all[:, 1])
    rp = EasyDict(x1=min_x, x2=max_x, y1=min_y, y2=max_y)
    log.debug("RectPosition:%s", rp)
    return rp


def get_scale_and_delta(ref_height, rect_value, img, delta_crop):
    if ref_height > 0:
        sc = rect_value['scale'] * 200 / ref_height
        img_sc = np.array(
            Image.fromarray(img).resize((int(img.shape[1] / sc), int(img.shape[0] / sc)), Image.BICUBIC))
        delta_x = delta_crop
        delta_y = delta_crop
    else:
        sc = 1.0
        img_sc = img
        delta_x = np.round(delta_crop * rect_value['scale'])
        delta_y = np.round(delta_crop * rect_value['scale'])
    log.debug('sc: %s', sc)
    log.debug('img_sc shape: %s', img_sc.shape)
    log.debug('delta_x: %s', delta_x)
    log.debug('delta_y: %s', delta_y)
    return sc, img_sc, delta_x, delta_y


def get_position(rp, sc):
    pos = EasyDict(
        x1=np.round(rp.x1 / sc),
        x2=np.round(rp.x2 / sc),
        y1=np.round(rp.y1 / sc),
        y2=np.round(rp.y2 / sc),
    )
    log.debug("pos %s", pos)
    return pos


def get_position_new(pos, delta_x, delta_y, img_sc):
    new = EasyDict(
        x1=np.round(max(1, pos.x1 - delta_x)),
        x2=np.round(min(img_sc.shape[1], pos.x2 + delta_x)),
        y1=max(1, pos.y1 - delta_y),
        y2=min(img_sc.shape[0], pos.y2 + delta_y),
    )
    log.debug("1st new %s", new)
    return new


def update_position_new(b_crop_isolated, rect, rect2, r_id, img_list, img_id, sc, pos, ref_height, rect_value, pos_new):
    if b_crop_isolated and len(rect) > 1:

        points2_all = []

        for r_id2, rect_value2 in enumerate(rect2):
            if r_id2 == r_id:
                continue
            points2 = rect_value2['points']
            if points2 is None or not points2:
                continue
            for f_p in points2:
                pp = [f_p['x'], f_p['y']]
                points2_all.append(pp)
        log.debug("points2_all: %s", points2_all)
        if points2_all:
            def max_index(d, idx):
                return np.argmax(d[idx])

            log.debug("img_list len :%s img_id:%d r_id:%s", len(img_list), img_id, r_id)
            points2_all = np.true_divide(points2_all, sc)
            d = points2_all[:, 0] - pos.x1
            idx = np.where(d < 0)[0]
            # log.debug("idx:%s max_index:%s", idx, max_index(d, idx))
            pos_x1other = None if not idx.any() else points2_all[idx[max_index(d, idx)], 0]

            d = points2_all[:, 1] - pos.y1
            idx = np.where(d < 0)[0]
            # pos_y1other = None if not idx.any() else points2_all[idx[max_index(d, idx)], 1]

            d = pos.x2 - points2_all[:, 0]
            idx = np.where(d < 0)[0]
            pos_x2other = None if not idx.any() else points2_all[idx[max_index(d, idx)], 0]

            d = pos.y2 - points2_all[:, 1]
            idx = np.where(d < 0)[0]
            # pos_y2other = None if not idx.any() else points2_all[idx[max_index(d, idx)], 1]

            if ref_height > 0:
                delta2 = ref_height / 200 * 10
            else:
                delta2 = rect_value['scale'] * 10

            if pos_x1other is not None:
                pos_new.x1 = np.round(max(pos_new.x1, pos_x1other + delta2))

            if pos_x2other is not None:
                pos_new.x2 = np.round(min(pos_new.x2, pos_x2other - delta2))
    pos_new.y1 = int(pos_new.y1)
    pos_new.y2 = int(pos_new.y2)
    pos_new.x1 = int(pos_new.x1)
    pos_new.x2 = int(pos_new.x2)

    log.debug("2nd new: %s", pos_new)
    return pos_new


def transform_annotation(points, pos, sc, rect_value):
    # transfer annotations
    log.debug("before transfer: %s pos.x1: %s pos.y1: %s sc: %s", points, pos.x1, pos.y1, sc)
    for pid in points:
        pid['x'] = pid['x'] / sc - pos.x1 + 1
        pid['y'] = pid['y'] / sc - pos.y1 + 1
    log.debug("after transfer: %s pos_new.x1: %s pos_new.y1: %s sc: %s", points, pos.x1, pos.y1, sc)
    rect_value['x1'] = rect_value['x1'] / sc - pos.x1 + 1
    rect_value['y1'] = rect_value['y1'] / sc - pos.y1 + 1
    rect_value['x2'] = rect_value['x2'] / sc - pos.x1 + 1
    rect_value['y2'] = rect_value['y2'] / sc - pos.y1 + 1

    rect_value['objpos_x'] = rect_value['objpos_x'] / sc - pos.x1 + 1
    rect_value['objpos_y'] = rect_value['objpos_y'] / sc - pos.y1 + 1


def get_annotation(annolists2, f_name, rect_value, image_size, num_crops):
    if not annolists2:
        obj = dict()
        obj['name'] = f_name
        obj['imgnum'] = 1
        obj['rect'] = rect_value
        obj['image_size'] = image_size
    else:
        num_crops = num_crops + 1
        obj = dict()
        obj['name'] = f_name
        obj['imgnum'] = num_crops
        obj['rect'] = rect_value
        obj['image_size'] = image_size
    return obj


def func_crop_data_train(option, img_list, img_ids, rectidxs):
    """
    crop train dataset
    """
    save_dir = option['saveDir']
    ref_height = option['refHeight']
    delta_crop = option['deltaCrop']
    b_crop_isolated = option['bCropIsolated']
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    annolists2 = []
    num_crops = 1
    num_images = 0

    for img_id, alv in enumerate(img_list):
        log.info("==> start img_id/all: %s/%s", img_id + 1, len(img_list))
        rect = alv['rect']
        rect2 = deepcopy(rect)
        name = alv['name']
        img = imread(os.path.join(option['imageDir'], name))
        log.debug("img :%s", alv)
        for r_id, rect_value in enumerate(rect):
            if r_id not in rectidxs[img_id]:
                continue
            log.info("==> start img_id: %s/%s r_id: %s/%s", img_id + 1, len(img_list), r_id + 1, len(rect))
            num_images = num_images + 1
            points = rect_value['points']
            if points is None or not points:
                continue
            rp = get_rect_position(points)
            sc, img_sc, delta_x, delta_y = get_scale_and_delta(ref_height, rect_value, img, delta_crop)
            pos = get_position(rp, sc)
            pos_new = get_position_new(pos, delta_x, delta_y, img_sc)
            pos_new = update_position_new(b_crop_isolated, rect, rect2, r_id, img_list, img_id, sc, pos, ref_height,
                                          rect_value, pos_new)
            img_crop = np.array([img_sc[pos_new.y1:pos_new.y2, pos_new.x1:pos_new.x2, 0],
                                 img_sc[pos_new.y1:pos_new.y2, pos_new.x1:pos_new.x2, 1],
                                 img_sc[pos_new.y1:pos_new.y2, pos_new.x1:pos_new.x2, 2]])

            img_crop = img_crop.transpose((1, 2, 0))
            # save image
            f_name = os.path.join(
                save_dir, 'im' + pad_zeros(img_ids[img_id], 5) + '_' + str(r_id) + '.png')
            f_name_t = os.path.join(
                save_dir, 'T_' + pad_zeros(img_ids[img_id], 5) + '_' + str(r_id) + '.json')
            log.debug('file name: %s', f_name)
            log.debug("image shape: %s", img_crop.shape)
            # noinspection PyTypeChecker
            imsave(f_name, img_crop)
            image_size = [img_crop.shape[0], img_crop.shape[1]]

            T = sc * np.array([[1, 0, pos_new.x1], [0, 1, pos_new.y1], [0, 0, 1]])
            with open(f_name_t, 'w') as f:
                f.write(json.dumps(T, default=json_default))

            transform_annotation(points, pos_new, sc, rect_value)
            anno = get_annotation(annolists2, f_name, rect_value, image_size, num_crops)
            num_crops = num_crops + 1
            annolists2.append(anno)
            log.info("==> finish img_id: %s/%s r_id: %s/%s ", img_id + 1, len(img_list), r_id + 1, len(rect))
        log.info("==> finish img_id: %s/%s", img_id + 1, len(img_list))
    return annolists2
