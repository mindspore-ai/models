# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""
python dataset.py
"""
import os
import numpy as np
from nibabel import load as load_nii

import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms as C2


def norm(image):
    """
    the normalization of image
    """
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def vox_generator(path, all_files, n_pos, n_neg, correction=False):
    """
    read a dataset of train

    Args:
        path(string): the path of dataset.
        all_files(list): the list of train path
        n_pos(int): the number of foreground
        n_neg(int): the number of background
        correction(bool): Whether to correct the image. Default: False

    Returns:
        data_norm, labels, centers
    """
    while True:
        for file in all_files:
            if correction:
                flair = load_nii(os.path.join(path, file, file + '_flair_corrected.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2_corrected.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1_corrected.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce_corrected.nii.gz')).get_data()
            else:

                flair = load_nii(os.path.join(path, file, file + '_flair.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce.nii.gz')).get_data()

            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])
            labels = load_nii(os.path.join(path, file, file + '_seg.nii.gz')).get_data()

            foreground = np.array(np.where(labels > 0))
            background = np.array(np.where((labels == 0) & (flair > 0)))

            foreground = foreground[:, np.random.permutation(foreground.shape[1])[:n_pos]]
            background = background[:, np.random.permutation(background.shape[1])[:n_neg]]

            centers = np.concatenate((foreground, background), axis=1)
            centers = centers[:, np.random.permutation(n_neg + n_pos)]

            yield data_norm, labels, centers


def vox_generator_test(data_path, all_files, correction=False):
    """
    read a dataset of test

    Args:
        data_path(string): the path of dataset.
        all_files(list): the list of test path
        correction(bool): Whether to correct the image. Default: False

    Returns:
        data, data_norm, labels
    """
    path = data_path
    while True:
        for file in all_files:
            if correction:
                flair = load_nii(os.path.join(path, file, file + '_flair_corrected.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2_corrected.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1_corrected.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce_corrected.nii.gz')).get_data()
            else:
                flair = load_nii(os.path.join(path, file, file + '_flair.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce.nii.gz')).get_data()

            data = np.array([flair, t2, t1, t1ce])
            data = np.transpose(data, axes=[1, 2, 3, 0])

            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])

            labels = load_nii(os.path.join(path, file, file + '_seg.nii.gz')).get_data()

            yield data, data_norm, labels


def get_patches_3d(data, labels, centers, hsize, wsize, csize, psize):
    """
    get 3d patches of data labels

    Args:
        data(array): the data of image.
        centers(array): the points of center
        hsize(int): the size of height
        wsize(int): the size of width
        csize(int): the size of depth

    Returns:
        patches_x, patches_y
    """
    patches_x, patches_y = [], []
    offset_p = (hsize - psize) // 2
    for i in range(len(centers[0])):
        h, w, c = centers[0, i], centers[1, i], centers[2, i]
        h_beg = min(max(0, h - hsize // 2), 240 - hsize)
        w_beg = min(max(0, w - wsize // 2), 240 - wsize)
        c_beg = min(max(0, c - csize // 2), 155 - csize)
        ph_beg = h_beg + offset_p
        pw_beg = w_beg + offset_p
        pc_beg = c_beg + offset_p
        vox = data[h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize, :]
        vox_labels = labels[ph_beg:ph_beg + psize, pw_beg:pw_beg + psize, pc_beg:pc_beg + psize]
        patches_x.append(vox)
        patches_y.append(vox_labels)
    return np.array(patches_x), np.array(patches_y)


def LoadData(data_path, train_path, HSIZE=38, WSIZE=38, CSIZE=38, PSIZE=12, correction=False):
    """
    load data of train dataset

    Args:
        data_path(string): the path of dataset.
        train_path(string): the path of train file.
        HSIZE(int): the size of height. Default: 38
        WSIZE(int): the size of width. Default: 38
        CSIZE(int): the size of depth. Default: 38
        PSIZE(int): the size of predict. Default: 12
        correction(bool): Whether to correct the image. Default: False

    Returns:
        flair_t2_nodes, t1_t1ce_nodes, label_batchs
    """
    files = []
    with open(train_path) as f:
        for line in f:
            files.append(line[:-1])
    print('%d training samples' % len(files))
    data_gen_train = vox_generator(data_path, all_files=files, n_pos=200, n_neg=200, correction=correction)
    for pi in range(len(files)):
        data, labels, centers = data_gen_train.__next__()
        data_batch, label_batch = get_patches_3d(data, labels,
                                                 centers, HSIZE,
                                                 WSIZE, CSIZE, PSIZE)
        data_batch = np.transpose(data_batch, axes=[0, 4, 1, 2, 3])
        flair_t2_node = data_batch[:, :2, :, :, :]
        t1_t1ce_node = data_batch[:, 2:, :, :, :]
        if pi == 0:
            flair_t2_nodes = flair_t2_node
            t1_t1ce_nodes = t1_t1ce_node
            label_batchs = label_batch
        else:
            flair_t2_nodes = np.concatenate((flair_t2_nodes, flair_t2_node), axis=0)
            t1_t1ce_nodes = np.concatenate((t1_t1ce_nodes, t1_t1ce_node), axis=0)
            label_batchs = np.concatenate((label_batchs, label_batch), axis=0)
    return flair_t2_nodes, t1_t1ce_nodes, label_batchs


class DatasetGenerator:
    """ DatasetGenerator """

    def __init__(self, flair_t2_node, t1_t1ce_node, label):
        self.flair_t2_node = flair_t2_node
        self.t1_t1ce_node = t1_t1ce_node
        self.label = label

    def __getitem__(self, index):
        return self.flair_t2_node[index], self.t1_t1ce_node[index], self.label[index]

    def __len__(self):
        return len(self.flair_t2_node)


def create_dataset(data_path="", train_path="", HSIZE=38, WSIZE=38, CSIZE=38, PSIZE=12, batch_size=2, correction=False,
                   target="Ascend", mindrecord_path="", use_mindrecord=False, group_size=1, device_id=0):
    """
    create a train dataset

    Args:
        data_path(string): the path of dataset Default: "".
        train_path(string): the path of train file Default: "".
        HSIZE(int): the size of height. Default: 38
        WSIZE(int): the size of width. Default: 38
        CSIZE(int): the size of depth. Default: 38
        PSIZE(int): the size of predict. Default: 12
        batch_size(int): the batch size of dataset. Default: 32
        correction(bool): Whether to correct the image. Default: False
        target(str): the device target. Default: Ascend
        mindrecord_path(str): the mindrecord path. used for GPU train
        group_size(int): the number of training device. Default: 1
        device_id(int): device target for train. Default: 0
        use_mindrecord(bool): if use mindrecord dataset to train
    Returns:
        dataset
    """
    if target == "Ascend":
        rank_size, rank_id = _get_rank_info()
    elif target == "GPU":
        rank_size, rank_id = group_size, device_id

    if use_mindrecord:
        if group_size == 1:
            rank_size = None
            rank_id = None

        train_loader = ds.MindDataset(mindrecord_path,
                                      columns_list=["flair_t2_node", "t1_t1ce_node", "label"],
                                      num_parallel_workers=8,
                                      num_shards=rank_size,
                                      shard_id=rank_id)
    else:
        flair_t2_node, t1_t1ce_node, label_batchs = LoadData(data_path, train_path, HSIZE,
                                                             WSIZE, CSIZE, PSIZE, correction)
        train_ds = DatasetGenerator(flair_t2_node=flair_t2_node, t1_t1ce_node=t1_t1ce_node, label=label_batchs)
        train_loader = ds.GeneratorDataset(train_ds, column_names=["flair_t2_node", "t1_t1ce_node", "label"],
                                           num_parallel_workers=8, shuffle=True, num_shards=rank_size, shard_id=rank_id)

    type_cast_float32_op = C2.TypeCast(mstype.float32)
    train_loader = train_loader.map(operations=type_cast_float32_op, input_columns="flair_t2_node",
                                    num_parallel_workers=8)
    train_loader = train_loader.map(operations=type_cast_float32_op, input_columns="t1_t1ce_node",
                                    num_parallel_workers=8)
    type_cast_int32_op = C2.TypeCast(mstype.int32)
    train_loader = train_loader.map(operations=type_cast_int32_op, input_columns="label", num_parallel_workers=8)
    train_loader = train_loader.batch(batch_size=batch_size, drop_remainder=True, num_parallel_workers=8)
    return train_loader

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
