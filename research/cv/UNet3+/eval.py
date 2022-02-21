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
'''eval of UNet3+'''
import datetime
import os

import mindspore.ops as ops
from mindspore import context
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net

from src.logger import get_logger
from src.dataset import create_Dataset
from src.models import UNet3Plus
from src.config import config as cfg

def copy_data_from_obs():
    '''copy_data_from_obs'''
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying test weights from obs to cache....")
        mox.file.copy_parallel(cfg.pretrain_path, 'cache/weight')
        cfg.logger.info("copying test weights finished....")
        cfg.pretrain_path = 'cache/weight/'

        cfg.logger.info("copying val dataset from obs to cache....")
        mox.file.copy_parallel(cfg.val_data_path, 'cache/val')
        cfg.logger.info("copying val dataset finished....")
        cfg.val_data_path = 'cache/val/'

def copy_data_to_obs():
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(cfg.save_dir, cfg.outer_path)
        cfg.logger.info("copying finished....")

def dice_coef(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(model_path):
    '''test'''
    model = UNet3Plus()
    model.set_train(False)
    cfg.logger.info('load test weights from %s', str(model_path))
    load_param_into_net(model, load_checkpoint(model_path))
    cfg.logger.info('loaded test weights from %s', str(model_path))

    val_dataset, _ = create_Dataset(cfg.val_data_path, 0, cfg.batch_size,\
                                    1, 0, shuffle=False)
    data_loader = val_dataset.create_dict_iterator()
    dices = AverageMeter()
    sigmoid = ops.Sigmoid()
    for _, data in enumerate(data_loader):
        output = sigmoid(model(data["image"])).asnumpy()
        dice = dice_coef(output, data["mask"].asnumpy())
        dices.update(dice, cfg.batch_size)
    cfg.logger.info("Final dices: %s", str(dices.avg))

if __name__ == '__main__':
    set_seed(1)
    cfg.save_dir = os.path.join(cfg.output_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not cfg.use_modelarts and not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target,
                        device_id=device_id, save_graphs=False)

    cfg.logger = get_logger(cfg.save_dir, "UNet3Plus", 0)
    cfg.logger.save_args(cfg)

    copy_data_from_obs()
    test(os.path.join(cfg.pretrain_path, cfg.ckpt_name))
    copy_data_to_obs()
