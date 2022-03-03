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
import sys

import numpy as np
import scipy.io as scio

from src.log import log
from src.tool.preprocess.crop import crop_data
from src.tool.preprocess.utils import json_default, mpii_mat2dict

np.set_printoptions(threshold=sys.maxsize)
basedir = os.path.abspath(os.path.dirname(__file__))


def preprocess_single(dataset_dir, dataset_name, save_dir=None, image_dir=None):
    """
    preprocess single person dataset

    Args:
        dataset_dir: path to dataset
        dataset_name: annotation file relative path
        save_dir: path to cropped images
        image_dir: path to original images
    """
    if save_dir is None:
        save_dir = os.path.join(dataset_dir, 'cropped')
    if image_dir is None:
        image_dir = os.path.join(dataset_dir, 'images')

    p = dict()

    p["bTrain"] = 1
    p["refHeight"] = 400
    p["deltaCrop"] = 130
    p["bSingle"] = 1
    p["bCropIsolated"] = 1
    p["bMulti"] = 0
    p["bObjposOffset"] = 1

    p["datasetDir"] = dataset_dir
    p["datasetName"] = os.path.join(p["datasetDir"], dataset_name)

    p["saveDir"] = save_dir
    p["imageDir"] = image_dir

    mat = scio.loadmat(p["datasetName"], struct_as_record=False)
    p["dataset"] = mpii_mat2dict(mat)

    img_list = crop_data(p)
    p['deltaCrop'] = 65
    p['bSingle'] = 0
    p['bCropIsolated'] = 0
    img_list2 = crop_data(p)

    img_list = img_list + img_list2

    img_list_full_name = os.path.join(p['saveDir'], 'annolist-full-h' + str(p['refHeight']) + '.json')
    with open(img_list_full_name, 'w') as f:
        f.write(json.dumps(img_list, default=json_default))

    prepare_training_data(img_list, p['saveDir'])


def prepare_training_data(img_list, saveDir):
    """
    generate final dataset file
    """
    zero_indexed_joints_ids = True
    pidxs = [0, 2, 4, 5, 7, 9, 12, 14, 16, 17, 19, 21, 22, 23]

    num_joints = len(pidxs)
    with open(os.path.join(basedir, 'parts.json'), 'r') as f:
        parts = json.load(f)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    num_images = len(img_list)
    channels = 3
    dataset = []

    for imgidx, imgv in enumerate(img_list):
        if imgidx + 1 % 100 == 0:
            log.info('processing image %s/%s', imgidx, num_images)

        filename = imgv['name']

        joints = np.zeros((num_joints, 3))
        all_joints = []
        rectv = imgv['rect']

        joint_list = get_anno_joints(rectv, pidxs, parts)

        n = 0
        for j in range(0, num_joints):
            jnt = joint_list[j, :]
            if not np.isnan(jnt[0]):
                joints[n, :] = np.concatenate([[j], jnt])
                n = n + 1

        joints = joints[:n, :]
        if zero_indexed_joints_ids:
            joints[:, 0] = joints[:, 0]
        else:
            joints[:, 0] = joints[:, 0] + 1
        all_joints.append(joints)

        entry = dict()
        entry['image'] = filename
        entry['size'] = np.concatenate([[channels], imgv['image_size']])
        entry['joints'] = all_joints
        dataset.append(entry)
    os.makedirs(saveDir, exist_ok=True)
    out_filename = os.path.join(saveDir, 'dataset.json')
    log.debug("Generated dataset definition file:%s", out_filename)
    with open(out_filename, 'w') as f:
        f.write(json.dumps(dataset, default=json_default))


def get_anno_joints(rect, pidxs, parts):
    """
    get annotation joints
    """
    num_joints = len(pidxs)
    joints = np.full((num_joints, 2), np.nan)
    points = rect['points']
    for j, pidx in enumerate(pidxs):
        annopoint_idxs = parts[pidx]['pos']
        assert annopoint_idxs[0] == annopoint_idxs[1]
        pt, _ = get_annopoint_by_id(points, annopoint_idxs[0])
        if pt is not None:
            joints[j, :] = np.array([pt['x'], pt['y']])
    return joints


def get_annopoint_by_id(points, idx):
    """
    get annotation point by id
    """
    point = None
    ind = None
    for i, v in enumerate(points):
        if v['id'] == idx:
            point = v
            ind = i
            return (point, ind)
    return (point, ind)
