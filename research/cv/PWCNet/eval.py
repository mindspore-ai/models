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
"""PWCNet eval."""
import os
import datetime
import warnings

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.pwcnet_model import PWCNet
from src.sintel import SintelTraining
from src.log import get_logger
from src.loss import MultiScaleEPE_PWC

from model_utils.config import config as cfg
from model_utils.device_adapter import get_device_id

warnings.filterwarnings("ignore")


def run_eval():
    """run eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=False)
    if cfg.device_target == "Ascend":
        context.set_context(device_id=get_device_id())

    cfg.local_rank = 0
    cfg.world_size = 1

    # logger
    cfg.outputs_dir = os.path.join(cfg.ckpt_path, datetime.datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S"))
    cfg.logger = get_logger(cfg.outputs_dir, cfg.local_rank)

    # Dataloader
    cfg.logger.info("start create dataloader")
    de_dataset, _ = SintelTraining(
        cfg.eval_dir,
        cfg.valid_augmentations,
        "final",
        "valid",
        cfg.val_batch_size,
        cfg.num_parallel_workers,
        cfg.local_rank,
        cfg.world_size,
    )
    de_dataloader = de_dataset.create_tuple_iterator(output_numpy=False, do_copy=False)
    # Show cfg
    cfg.logger.save_args(cfg)
    cfg.logger.info("end create dataloader")

    # backbone and loss
    cfg.logger.important_info("start create network")

    network = PWCNet()
    criterion = MultiScaleEPE_PWC()

    # load pretrain model
    if os.path.isfile(cfg.pretrained):
        param_dict = load_checkpoint(cfg.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith(
                "moment1." or "moment2" or "global_step" or "beta1_power" or "beta2_power" or "learning_rate"
            ):
                continue
            elif key.startswith("network."):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        cfg.logger.info("load model %s success.", cfg.pretrained)

    network.set_train(False)

    cfg.logger.important_info("====start eval====")
    validation_loss = 0
    sum_num = 0
    for _, data in enumerate(de_dataloader):
        output = network(data[0], data[1], training=False)
        loss = criterion(output, data[2], training=False)
        validation_loss += loss
        sum_num += 1
    print("EPE: ", validation_loss / sum_num)

    cfg.logger.important_info("====eval end====")


if __name__ == "__main__":
    run_eval()
