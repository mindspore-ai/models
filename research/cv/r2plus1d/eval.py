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
'''eval.py'''
import os
import datetime

import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net

from src.logger import get_logger
from src.dataset import create_VideoDataset
from src.models import get_r2plus1d_model
from src.config import config as cfg

def copy_data_from_obs():
    '''copy_data_from_obs'''
    if cfg.use_modelarts:
        import moxing as mox
        import zipfile
        cfg.logger.info("copying validation weights from obs to cache....")
        mox.file.copy_parallel(cfg.eval_ckpt_path, 'cache/weight')
        cfg.logger.info("copying validation weights finished....")
        cfg.eval_ckpt_path = 'cache/weight/'

        cfg.logger.info("copying dataset from obs to cache....")
        mox.file.copy_parallel(cfg.dataset_root_path, 'cache/dataset')
        cfg.logger.info("copying dataset finished....")
        cfg.dataset_root_path = 'cache/dataset/'
        cfg.logger.info("starting unzip file to cache....")
        zFile = zipfile.ZipFile(os.path.join(cfg.dataset_root_path, cfg.pack_file_name), "r")
        for fileM in zFile.namelist():
            zFile.extract(fileM, cfg.dataset_root_path)
        zFile.close()
        cfg.dataset_root_path = os.path.join(cfg.dataset_root_path, cfg.pack_file_name.split(".")[0])
        cfg.logger.info("unzip finished....")

def copy_data_to_obs():
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(cfg.save_dir, cfg.outer_path)
        cfg.logger.info("copying finished....")

def test():
    '''
    main test function
    '''
    f_model = get_r2plus1d_model(cfg.num_classes, cfg.layer_num)
    f_model.set_train(False)
    eval_ckpt_path = cfg.eval_ckpt
    cfg.logger.info('load validation weights from %s', str(eval_ckpt_path))
    load_param_into_net(f_model, load_checkpoint(eval_ckpt_path))
    cfg.logger.info('loaded validation weights from %s', str(eval_ckpt_path))

    val_dataloader, cfg.steps_per_epoch = create_VideoDataset(cfg.dataset_root_path, cfg.dataset_name, \
                        mode=cfg.val_mode, clip_len=16, batch_size=cfg.batch_size, \
                        device_num=1, rank=0, shuffle=False)
    cfg.logger.info("cfg.steps_per_epoch: %s", str(cfg.steps_per_epoch))

    # optimizer
    optimizer = nn.SGD(params=f_model.trainable_params(), momentum=cfg.momentum,
                       learning_rate=0.001, weight_decay=cfg.weight_decay)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='sum')
    model = Model(f_model, loss_fn, optimizer, amp_level="auto", metrics={'top_1_accuracy', 'top_5_accuracy'})
    result = model.eval(val_dataloader, dataset_sink_mode=False)
    cfg.logger.info('Final Accuracy: %s', str(result))
    cfg.logger.info("validation finished....")

if __name__ == '__main__':
    set_seed(1)
    cfg.save_dir = os.path.join(cfg.output_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not cfg.use_modelarts and not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target,
                        device_id=device_id, save_graphs=False)
    cfg.logger = get_logger(cfg.save_dir, "R2plus1D", 0)
    cfg.logger.save_args(cfg)

    copy_data_from_obs()
    test()
    copy_data_to_obs()
    cfg.logger.info('All task finished!')
