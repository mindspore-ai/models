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

"""Cycle GAN test."""

import os
import datetime
import argparse
import moxing as mox

from mindspore import Tensor
from src.models.cycle_gan import get_generator
from src.utils.args import get_args
from src.dataset.cyclegan_dataset import create_dataset
from src.utils.reporter import Reporter
from src.utils.tools import save_image, load_ckpt

parser = argparse.ArgumentParser(description='CycleGAN Evaling Args')
parser.add_argument("--modelarts_FLAG", type=bool, default=True, help="use modelarts or not")
parser.add_argument('--dataroot', type=str, default='./data/', help="data path on obs")
parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset/")
parser.add_argument('--G_A_ckpt', type=str,
                    default='/cache/dataset/model/cyclegan_GB_ascend_v170_apple2orange_research_cv.ckpt')
parser.add_argument('--G_B_ckpt', type=str,
                    default='/cache/dataset/model/cyclegan_GA_ascend_v170_apple2orange_research_cv.ckpt')
parser.add_argument("--val_dataset", type=str, default="/cache/dataset/apple2orange/")
parser.add_argument("--val_url", type=str, default="./output/val_save")
args_opt = parser.parse_args()


def obs_data2modelarts(cfg):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(cfg.dataroot, cfg.modelarts_data_dir))
    mox.file.copy_parallel(src_url=cfg.dataroot, dst_url=cfg.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(cfg.modelarts_data_dir)
    print("===>>>before Files:", files)


def modelarts_result2obs(cfg):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=cfg.modelarts_result_dir, dst_url=cfg.outputs_dir)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(cfg.modelarts_result_dir,
                                                                                  cfg.outputs_dir))


def predict(cfg):
    """Predict function."""
    args = get_args("predict")
    if cfg.modelarts_FLAG:
        obs_data2modelarts(cfg)
        args.dataroot = cfg.val_dataset
        args.G_A_ckpt = cfg.G_A_ckpt
        args.G_B_ckpt = cfg.G_B_ckpt
    G_A = get_generator(args)
    G_B = get_generator(args)
    G_A.set_train(True)
    G_B.set_train(True)
    load_ckpt(args, G_A, G_B)
    imgs_out = os.path.join(args.outputs_dir, "predict")
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)
    if not os.path.exists(os.path.join(imgs_out, "fake_A")):
        os.makedirs(os.path.join(imgs_out, "fake_A"))
    if not os.path.exists(os.path.join(imgs_out, "fake_B")):
        os.makedirs(os.path.join(imgs_out, "fake_B"))
    args.data_dir = 'testA'
    ds = create_dataset(args)
    reporter = Reporter(args)
    reporter.start_predict("A to B")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_A = Tensor(data["image"])
        path_A = str(data["image_name"][0], encoding="utf-8")
        path_B = path_A[0:-4] + "_fake_B.jpg"
        fake_B = G_A(img_A)
        save_image(fake_B, os.path.join(imgs_out, "fake_B", path_B))
        save_image(img_A, os.path.join(imgs_out, "fake_B", path_A))
    reporter.info('save fake_B at %s', os.path.join(imgs_out, "fake_B", path_A))
    reporter.end_predict()
    args.data_dir = 'testB'
    ds = create_dataset(args)
    reporter.dataset_size = args.dataset_size
    reporter.start_predict("B to A")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_B = Tensor(data["image"])
        path_B = str(data["image_name"][0], encoding="utf-8")
        path_A = path_B[0:-4] + "_fake_A.jpg"
        fake_A = G_B(img_B)
        save_image(fake_A, os.path.join(imgs_out, "fake_A", path_A))
        save_image(img_B, os.path.join(imgs_out, "fake_A", path_B))
    reporter.info('save fake_A at %s', os.path.join(imgs_out, "fake_A", path_B))
    reporter.end_predict()

    if cfg.modelarts_FLAG:
        mox.file.copy_parallel(src_url=os.path.join(imgs_out, "fake_A"), dst_url=cfg.val_url)
        print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(os.path.join(imgs_out, "fake_A"),
                                                                                      cfg.val_url))
        mox.file.copy_parallel(src_url=os.path.join(imgs_out, "fake_B"), dst_url=cfg.val_url)
        print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(os.path.join(imgs_out, "fake_B"),
                                                                                      cfg.val_url))

if __name__ == "__main__":
    predict(cfg=args_opt)
    