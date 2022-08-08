# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
######################## train lenet example ########################
train lenet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import numpy as np
from mindspore import dtype, Tensor, context, export, load_checkpoint

from src.config.defaults import assert_and_infer_cfg
from src.utils.parser import load_config, parse_args
from src.utils import logging
from src.models.video_model_builder import SlowFast

def run_export():
    "Entrance method"
    # setup arguments
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    # setup logger
    logger = logging.get_logger(__name__)
    logging.setup_logging()
    logger.info(cfg)

    # setup context
    context.set_context(device_id=0, mode=context.GRAPH_MODE, device_target="Ascend")
    # build network
    network = SlowFast(cfg).to_float(dtype.float32).set_train(False)
    # load ckpt
    load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, network)

    # generate dummy input
    slowpath = Tensor(np.zeros((cfg.TEST.BATCH_SIZE,
                                3,
                                cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA,
                                cfg.DATA.TEST_SCALE_HEIGHT,
                                cfg.DATA.TEST_SCALE_WIDTH), dtype=np.float16))
    fastpath = Tensor(np.zeros((cfg.TEST.BATCH_SIZE,
                                3,
                                cfg.DATA.NUM_FRAMES,
                                cfg.DATA.TEST_SCALE_HEIGHT,
                                cfg.DATA.TEST_SCALE_WIDTH), dtype=np.float16))
    boxes = Tensor(np.zeros((cfg.TEST.BATCH_SIZE,
                             cfg.DATA.MAX_NUM_BOXES_PER_FRAME,
                             4), dtype=np.float32))
    export(network, slowpath, fastpath, boxes, file_name='slowfast', file_format='MINDIR')
    print('export successfully!')

if __name__ == "__main__":
    run_export()
