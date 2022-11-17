# -*- encoding: utf-8 -*-
"""
  @Project: LSTM
  @File: data_process.py
  @Author: Joy
  @Created Time: 2021-11-12 19:59
"""
import argparse
import os
import time

import moxing as mox
import numpy as np
import mindspore.dataset as ds
from mindspore import context

from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

parser = argparse.ArgumentParser(description='Natural Language Processing')

# ModelArts config
parser.add_argument("--enable_modelarts", type=bool, default=True, help="whether training on modelarts, default: True")
parser.add_argument("--data_url", type=str, default="", help="dataset url for obs")
parser.add_argument("--checkpoint_url", type=str, default="", help="checkpoint url for obs")
parser.add_argument("--train_url", type=str, default="", help="training output url for obs")
parser.add_argument("--data_path", type=str, default="/cache/data", help="dataset path for local")
parser.add_argument("--load_path", type=str, default="/cache/checkpoint", help="dataset path for local")
parser.add_argument("--output_path", type=str, default="/cache/train", help="training output path for local")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/",
                    help="the path where pre-trained checkpoint file path")
parser.add_argument("--checkpoint_file", type=str, default="./checkpoint/lstm-20_390.ckpt",
                    help="the path where pre-trained checkpoint file name")
parser.add_argument("--device_target", type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("--enable_profiling", type=bool, default=False, help="whether enable modelarts profiling")

# LSTM config
parser.add_argument("--num_classes", type=int, default=2, help="output class num")
parser.add_argument("--num_hiddens", type=int, default=128, help="number of hidden unit per layer")
parser.add_argument("--num_layers", type=int, default=2, help="number of network layer")
parser.add_argument("--learning_rate", type=float, default=0.1, help="static learning rate")
parser.add_argument("--dynamic_lr", type=bool, default=False, help="dynamic learning rate")
parser.add_argument("--lr_init", type=float, default=0.05,
                    help="initial learning rate, effective when enable dynamic_lr")
parser.add_argument("--lr_end", type=float, default=0.01, help="end learning rate, effective when enable dynamic_lr")
parser.add_argument("--lr_max", type=float, default=0.1, help="maximum learning rate, effective when enable dynamic_lr")
parser.add_argument("--lr_adjust_epoch", type=int, default=6,
                    help="the epoch interval of adjusting learning rate, effective when enable dynamic_lr")
parser.add_argument("--warmup_epochs", type=int, default=1,
                    help="the epoch interval of warmup, effective when enable dynamic_lr")
parser.add_argument("--momentum", type=float, default=0.9, help="")
parser.add_argument("--num_epochs", type=int, default=20, help="")
parser.add_argument("--batch_size", type=int, default=64, help="")
parser.add_argument("--embed_size", type=int, default=300, help="")
parser.add_argument("--bidirectional", type=bool, default=True, help="whether enable bidirectional LSTM network")
parser.add_argument("--save_checkpoint_steps", type=int, default=7800, help="")
parser.add_argument("--keep_checkpoint_max", type=int, default=10, help="")

# train config
parser.add_argument("--preprocess", type=str, default='false', help="whether to preprocess data")
parser.add_argument("--preprocess_path", type=str, default="./preprocess",
                    help="path where the pre-process data is stored, "
                         "if preprocess set as 'false', you need prepared preprocessed data under data_url")
parser.add_argument("--aclImdb_zip_path", type=str, default="./aclImdb_v1.tar.gz", help="path where the dataset zip")
parser.add_argument("--aclImdb_path", type=str, default="./aclImdb", help="path where the dataset is stored")
parser.add_argument("--glove_path", type=str, default="./glove", help="path where the GloVe is stored")
parser.add_argument("--ckpt_path", type=str, default="./ckpt_lstm/",
                    help="the path to save the checkpoint file")
parser.add_argument("--pre_trained", type=str, default="", help="the pretrained checkpoint file path")
parser.add_argument("--device_num", type=int, default=1, help="the number of using devices")
parser.add_argument("--distribute", type=bool, default=False, help="enable when training with multi-devices")
parser.add_argument("--enable_graph_kernel", type=bool, default=True, help="whether accelerate by graph kernel")

# export config
parser.add_argument("--ckpt_file", type=str, default="./ckpt_lstm/lstm-20_390.ckpt", help="the export ckpt file name")
parser.add_argument("--device_id", type=int, default=0, help="")
parser.add_argument("--file_name", type=str, default="./lstm", help="the export air file name")
parser.add_argument("--file_format", type=str, default="AIR", help="the export file format")

# LSTM Postprocess config
parser.add_argument("--label_dir", type=str, default="", help="")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="")

# Preprocess config
parser.add_argument("--result_path", type=str, default="./preprocess_Result/", help="")

config = parser.parse_args()

_global_sync_count = 0


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    :param from_path: source path
    :param to_path: target path
    :return: no return
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Finish sync data from {} to {}.".format(from_path, to_path))


def download_data():
    """
    sync data from data_url, train_url to data_path, output_path
    :return: no return
    """
    if config.enable_modelarts:
        if config.data_url:
            if not os.path.isdir(config.data_path):
                os.makedirs(config.data_path)
                sync_data(config.data_url, config.data_path)
                print("Dataset downloaded: ", os.listdir(config.data_path))
        if config.checkpoint_url:
            if not os.path.isdir(config.load_path):
                os.makedirs(config.load_path)
                sync_data(config.checkpoint_url, config.load_path)
                print("Preload downloaded: ", os.listdir(config.load_path))
        if config.train_url:
            if not os.path.isdir(config.output_path):
                os.makedirs(config.output_path)
            sync_data(config.train_url, config.output_path)
            print("Workspace downloaded: ", os.listdir(config.output_path))

        context.set_context(save_graphs_path=os.path.join(config.output_path, str(get_rank_id())))
        config.device_num = get_device_num()
        config.device_id = get_device_id()
        # create output dir
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)


def upload_data():
    """
    sync data from output_path to train_url
    :return: no return
    """
    if config.enable_modelarts:
        if config.train_url:
            print("Start copy data to output directory.")
            sync_data(config.output_path, config.train_url)
            print("Copy data to output directory finished.")


def modelarts_preprocess():
    """
    add path prefix, modify parameter and sync data
    :return: no return
    """
    print("============== Starting ModelArts Preprocess ==============")
    config.aclImdb_path = os.path.join(config.data_path, config.aclImdb_path)
    config.aclImdb_zip_path = os.path.join(config.data_path, config.aclImdb_zip_path)
    config.glove_path = os.path.join(config.data_path, config.glove_path)

    config.file_name = os.path.join(config.output_path, config.file_name)
    config.result_path = os.path.join(config.output_path, config.result_path)

    if config.preprocess == 'true':
        config.preprocess_path = os.path.join(config.output_path, config.preprocess_path)
    else:
        config.preprocess_path = os.path.join(config.data_path, config.preprocess_path)

    # download data from obs
    download_data()
    print("============== ModelArts Preprocess finish ==============")


def modelarts_postprocess():
    """
    convert lstm model to AIR format, sync data
    :return: no return
    """
    print("============== Starting ModelArts Postprocess ==============")
    # upload data to obs
    upload_data()
    print("============== ModelArts Postprocess finish ==============")


def lstm_create_dataset(data_home, batch_size, repeat_num=1, training=True, device_num=1, rank=0):
    """Data operations."""
    ds.config.set_seed(1)
    data_dir = os.path.join(data_home, "aclImdb_train.mindrecord0")
    if not training:
        data_dir = os.path.join(data_home, "aclImdb_test.mindrecord0")

    data_set = ds.MindDataset(data_dir, columns_list=["feature", "label"], num_parallel_workers=4,
                              num_shards=device_num, shard_id=rank)

    # apply map operations on images
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
    data_set = data_set.repeat(count=repeat_num)

    return data_set


def covert_to_bin():
    """
    save dataset with bin format
    :return: no return
    """
    dataset = lstm_create_dataset(config.preprocess_path, config.batch_size, training=False)
    img_path = os.path.join(config.result_path, "00_data")
    os.makedirs(img_path)
    label_list = []
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "LSTM_data_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        data['feature'].tofile(file_path)
        label_list.append(data['label'])

        print('processed {}.'.format(file_name))

    # save as npy
    np.save(config.result_path + "label_ids.npy", label_list)

    # save as txt
    sentence_labels = np.array(label_list)
    sentence_labels = sentence_labels.reshape(-1, 1).astype(np.int32)
    np.savetxt(config.result_path + "labels.txt", sentence_labels)

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == "__main__":
    modelarts_preprocess()
    covert_to_bin()
    modelarts_postprocess()
