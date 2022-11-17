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
"""split for CPU dataset"""
import os
import shutil
import multiprocessing
import mindspore as ms
import mindspore.dataset as ds


def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers


def create_dataset(dataset_path, do_train, batch_size=32, train_image_size=224, eval_image_size=224,
                   enable_cache=False, cache_session_id=None):
    """
       create a train or eval flower dataset for vgg16

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """

    ds.config.set_prefetch_size(64)
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=get_num_parallel_workers(12), shuffle=True)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            ds.vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ds.vision.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        trans = [
            ds.vision.Decode(),
            ds.vision.Resize(256),
            ds.vision.CenterCrop(eval_image_size)
        ]
    trans_norm = [ds.vision.Normalize(mean=mean, std=std), ds.vision.HWC2CHW()]

    type_cast_op = ds.transforms.TypeCast(ms.int32)
    trans_work_num = 24
    data_set = data_set.map(operations=trans, input_columns="image",
                            num_parallel_workers=get_num_parallel_workers(trans_work_num))
    data_set = data_set.map(operations=trans_norm, input_columns="image",
                            num_parallel_workers=get_num_parallel_workers(12))
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                num_parallel_workers=get_num_parallel_workers(12),
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                num_parallel_workers=get_num_parallel_workers(12))

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


def generate_data(path="./"):
    dirs = []
    abs_path = None
    for abs_path, j, _ in os.walk(path):
        print("abs_path:", abs_path)
        if j:
            dirs.append(j)
    print(dirs)

    train_folder = os.path.exists(path + 'train')
    if not train_folder:
        os.makedirs(path + 'train')
    test_folder = os.path.exists(path + 'test')
    if not test_folder:
        os.makedirs(path + 'test')

    for class_dir in dirs[0]:
        print("path", path)
        print("dir", class_dir)
        files = os.listdir(path + class_dir)
        train_set = files[: int(len(files) * 0.8)]
        test_set = files[int(len(files) * 0.8):]
        for file in train_set:
            file_path = path + "train/" + class_dir + "/"
            folder = os.path.exists(file_path)
            if not folder:
                os.makedirs(file_path)
            src_file = path + class_dir + "/" + file
            print("src_file:", src_file)
            dst_file = file_path + file
            print("dst_file:", dst_file)
            shutil.copyfile(src_file, dst_file)

        for file in test_set:
            file_path = path + "test/" + class_dir + "/"
            folder = os.path.exists(file_path)
            if not folder:
                os.makedirs(file_path)
            src_file = path + class_dir + "/" + file
            dst_file = file_path + file
            shutil.copyfile(src_file, dst_file)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", help="the path of dataset to be split")
    args = parser.parse_args()

    generate_data(path=args.split_path)

    create_dataset(dataset_path=args.split_path + "train/", do_train=True, batch_size=32, train_image_size=224,
                   eval_image_size=224,
                   enable_cache=False, cache_session_id=None)


if __name__ == '__main__':
    main()
