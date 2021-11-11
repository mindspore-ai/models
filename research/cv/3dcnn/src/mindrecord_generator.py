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
""" convert mri data to mindrecord """

import os
import argparse
import numpy as np
from nibabel import load as load_nii
from mindspore.mindrecord import FileWriter
from mindspore.common import set_seed

set_seed(1)


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


def generate_MRImindrecord(mindrecord_file, data_path, files,
                           HSIZE=38, WSIZE=38, CSIZE=38, PSIZE=12,
                           SAMPLE_NUM=400, correction=False):
    """
        create mindrecord data of train dataset

        Args:
            data_path(string): the path of dataset.
            train_path(string): the path of train file.
            HSIZE(int): the size of height. Default: 38
            WSIZE(int): the size of width. Default: 38
            CSIZE(int): the size of depth. Default: 38
            PSIZE(int): the size of predict. Default: 12
            correction(bool): Whether to correct the image. Default: False
    """
    writer = FileWriter(file_name=mindrecord_file, shard_num=4)
    file_schema = {
        "flair_t2_node": {"type": "float32", "shape": [2, 38, 38, 38]},
        "t1_t1ce_node": {"type": "float32", "shape": [2, 38, 38, 38]},
        "label": {"type": "float32", "shape": [12, 12, 12]}
    }
    writer.add_schema(file_schema)

    data_gen_train = vox_generator(data_path, all_files=files,
                                   n_pos=SAMPLE_NUM, n_neg=SAMPLE_NUM,
                                   correction=correction)
    for pi in range(len(files)):
        file = files[pi]

        data, labels, centers = data_gen_train.__next__()
        data_batch, label_batch = get_patches_3d(data, labels,
                                                 centers, HSIZE,
                                                 WSIZE, CSIZE, PSIZE)
        data_batch = np.transpose(data_batch, axes=[0, 4, 1, 2, 3])

        data_list = []
        for n in range(len(data_batch)):
            nb_data = {
                "flair_t2_node": data_batch[n][:2, :, :, :],
                "t1_t1ce_node": data_batch[n][2:, :, :, :],
                "label": label_batch[n]
            }

            data_list.append(nb_data)
            if (n + 1) % 100 == 0:
                print("{} {}-{} write successfully".format(file, str((n + 1) - 100), str(n)))
                writer.write_raw_data(data_list)
                data_list = []
        print("{} write all successfully".format(file))
    writer.commit()
    print("all file write all successfully")

def parse_inputs():
    """ init arguments """
    parser = argparse.ArgumentParser(description='conver data to mindrecord')

    parser.add_argument('-r', '--data-path', dest='data_path', type=str,
                        default='../MICCAI_BraTS17_Data_Training/HGG/')
    parser.add_argument('--train-path', dest='train_path', type=str,
                        default='./train.txt',
                        help='path to train.txt')
    parser.add_argument('--mindrecord-path', dest='mindrecord_path', type=str,
                        default='../dataset/',
                        help='path to mindrecord')
    parser.add_argument('--sample-num', dest='sample_num', type=int,
                        default='400',
                        help='negative and positive sample number')
    my_args = parser.parse_args()

    return my_args

args = parse_inputs()

if __name__ == '__main__':
    data_path_ = args.data_path
    train_path_ = args.train_path
    mindrecord_file_ = args.mindrecord_path + "mri2017.mindrecord_"
    sample_num = args.sample_num

    if sample_num < 0:
        print("sample number must bigger than 0, recommend 200 400 600 ...")
        exit(-1)

    all_files_ = []
    with open(train_path_) as f:
        for line in f:
            all_files_.append(line[:-1])
    print('%d training samples' % len(all_files_))

    generate_MRImindrecord(data_path=data_path_, files=all_files_, mindrecord_file=mindrecord_file_,
                           HSIZE=38, WSIZE=38, CSIZE=38,
                           PSIZE=12, SAMPLE_NUM=400, correction=True)
