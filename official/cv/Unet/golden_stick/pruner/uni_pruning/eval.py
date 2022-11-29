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

import logging
import os
import json
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.pruner.uni_pruning import UniPruner

from src.data_loader import create_dataset, create_multi_class_dataset
from src.unet_medical import UNetMedical
from src.utils import UnetEval, TempLoss, dice_coeff
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

@moxing_wrapper()
def test_net(data_dir,
             ckpt_path,
             cross_valid_ind=1):
    net = UNetMedical(n_channels=config.num_channels, n_classes=config.num_classes)

    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    net = UnetEval(net, eval_activate=config.eval_activate.lower())
    net.set_train(False)

    # pruner network
    input_size = [config.batch_size, config.num_channels, config.image_size[0], config.image_size[1]]
    algo = UniPruner({"exp_name": config.exp_name,
                      "frequency": config.frequency,
                      "target_sparsity": 1 - config.prune_rate,
                      "pruning_step": config.pruning_step,
                      "filter_lower_threshold": config.filter_lower_threshold,
                      "input_size": input_size,
                      "output_path": config.output_path,
                      "prune_flag": config.prune_flag,
                      "rank": 1,
                      "device_target": config.device_target})
    algo.apply(net)

    if config.mask_path is not None:
        mask_path = os.path.realpath(config.mask_path)
        if not os.path.exists(config.mask_path):
            raise ValueError("The mask json file: {} does not exist, please check whether the 'config.mask' is "
                             "correct.".format(mask_path))
        with open(config.mask_path, 'r', encoding='utf8') as json_fp:
            mask = json.load(json_fp)
        tag = 'pruned'
    else:
        mask = None
        tag = 'original'
    algo.prune_by_mask(net, mask, config, tag)

    if hasattr(config, "dataset") and config.dataset != "ISBI":
        split = config.split if hasattr(config, "split") else 0.8
        valid_dataset = create_multi_class_dataset(data_dir, config.image_size, 1, 1,
                                                   num_classes=config.num_classes, is_train=False,
                                                   eval_resize=config.eval_resize, split=split, shuffle=False)
    else:
        _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                          do_crop=config.crop, img_size=config.image_size)
    model = Model(net, loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(show_eval=config.show_eval)})

    print("============== Starting Evaluating ============")
    eval_score = model.eval(valid_dataset, dataset_sink_mode=False)["dice_coeff"]
    print("============== Cross valid dice coeff is:", eval_score[0])
    print("============== Cross valid IOU is:", eval_score[1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    assert config.device_target == "GPU"
    test_net(data_dir=config.data_path,
             ckpt_path=config.checkpoint_file_path,
             cross_valid_ind=config.cross_valid_ind)
