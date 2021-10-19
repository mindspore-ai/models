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
"""
This is the boot file for ModelArts platform.
Firstly, the train datasets are copied from obs to ModelArts.
Then, the string of train shell command is concated and using 'os.system()' to execute
"""

import os
import datetime
import tqdm
import moxing as mox

import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import export
from mindspore.common import set_seed
from modelarts.models import STGANModel
from modelarts.dataset import CelebADataLoader
from modelarts.utils import get_args

set_seed(1)

print(os.system('env'))


def obs_data2modelarts(args):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(args.data_url, args.modelarts_data_dir))
    mox.file.copy_parallel(src_url=args.data_url, dst_url=args.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(args.modelarts_data_dir)
    print("===>>>Files:", files)
    files = os.listdir(args.modelarts_data_dir + "/anno")
    print("===>>>anno:", files)
    files = os.listdir(args.modelarts_data_dir + "/image")
    print("===>>>image:", files)
    if not mox.file.exists(args.obs_result_dir):
        mox.file.make_dirs(args.obs_result_dir)
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(args.obs_result_dir, args.modelarts_result_dir))
    mox.file.copy_parallel(src_url=args.obs_result_dir, dst_url=args.modelarts_result_dir)
    files = os.listdir(args.modelarts_result_dir)
    print("===>>>Files:", files)

def modelarts_result2obs(args):
    """
    Copy result data from modelarts to obs.
    """
    obs_result_dir = args.obs_result_dir
    if not mox.file.exists(obs_result_dir):
        mox.file.make_dirs(obs_result_dir)
    mox.file.copy_parallel(src_url=args.modelarts_result_dir, dst_url=obs_result_dir)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(args.modelarts_result_dir,
                                                                                  obs_result_dir))
    files = os.listdir()
    print("===>>>current Files:", files)
    mox.file.copy(src_url='STGAN_G_model.air',
                  dst_url=os.path.join(obs_result_dir, args.experiment_name, 'STGAN_G_model.air'))

def export_AIR(args):
    """start modelarts export"""
    args1 = args
    args1.phase = "test"
    args1.isTrain = False
    args1.continue_train = False
    args1.ckpt_path = os.path.join(args.outputs_dir, args.experiment_name, "ckpt/latest_G.ckpt")
    net = STGANModel(args1)

    input_shp = [1, 3, args.image_size, args.image_size]
    input_shp_2 = [1, len(args.attrs)]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    input_array_2 = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp_2).astype(np.float32))
    export(net.netG, input_array, input_array_2, file_name="STGAN_G_model", file_format='AIR')

def train():
    """Train Function"""
    args = get_args("train")


    ## copy dataset from obs to modelarts
    if not args.continue_train:
        args.obs_result_dir = args.train_url
    obs_data2modelarts(args)

    args.dataroot = args.modelarts_data_dir
    args.outputs_dir = args.modelarts_result_dir
    args.attrs = args.modelarts_attrs.split(',')

    print(args)
    print('\n\n=============== start training ===============\n\n')

    # Get DataLoader
    data_loader = CelebADataLoader(args.dataroot,
                                   mode=args.phase,
                                   selected_attrs=args.attrs,
                                   batch_size=args.batch_size,
                                   image_size=args.image_size,
                                   device_num=args.device_num)
    iter_per_epoch = len(data_loader)
    args.dataset_size = iter_per_epoch

    # Get STGAN MODEL
    model = STGANModel(args)
    it_count = 0

    for _ in tqdm.trange(args.n_epochs, desc='Epoch Loop'):
        for _ in tqdm.trange(iter_per_epoch, desc='Inner Epoch Loop'):
            if model.current_iteration > it_count:
                it_count += 1
                continue
            try:
                # training model
                data = next(data_loader.train_loader)
                model.set_input(data)
                model.optimize_parameters()

                # saving model
                if (it_count + 1) % args.save_freq == 0:
                    model.save_networks()

                # sampling
                if (it_count + 1) % args.sample_freq == 0:
                    model.eval(data_loader)

            except KeyboardInterrupt:
                logger.info('You have entered CTRL+C.. Wait to finalize')
                model.save_networks()

            it_count += 1
            model.current_iteration = it_count

    model.save_networks()

    ## start export air
    export_AIR(args)
    ## copy result from modelarts to obs
    modelarts_result2obs(args)
    print('\n\n=============== finish training ===============\n\n')


if __name__ == '__main__':
    train()
