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

"""Eval."""
import os
import numpy as np
from mindspore import dtype, context, load_checkpoint

from src.utils.meters import AVAMeter
from src.config.defaults import assert_and_infer_cfg
from src.utils.parser import load_config, parse_args
from src.utils import logging
from src.datasets.build import build_dataset
from src.models.video_model_builder import SlowFast


def run_eval():
    """Entrance method."""
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    # setup logger
    logger = logging.get_logger(__name__)
    logging.setup_logging()
    logger.info(cfg)
    # setup context
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(device_id=device_id, mode=context.GRAPH_MODE, device_target=args.device_target)
    # build dataset
    dataset = build_dataset(cfg, "test")
    # build network
    network = SlowFast(cfg).to_float(dtype.float32).set_train(False)
    # load ckpt
    load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, network)
    # setup meter
    test_meter = AVAMeter(dataset.get_dataset_size(), cfg, mode="test")
    test_meter.iter_tic()

    # start eval
    for cur_iter, (slowpath, fastpath, boxes, _, ori_boxes,
                   metadata, mask) in enumerate(dataset.create_tuple_iterator()):
        test_meter.data_toc()
        preds = network(slowpath, fastpath, boxes)
        test_meter.iter_toc()
        # process output
        preds = preds.asnumpy().reshape(mask.shape + (cfg.MODEL.NUM_CLASSES,))
        mask = mask.asnumpy().astype(bool)
        preds = preds[mask]
        padded_idx = np.tile(np.arange(mask.shape[0]).reshape(
            (-1, 1, 1)), (1, mask.shape[1], 1))
        ori_boxes = np.concatenate(
            (padded_idx, ori_boxes.asnumpy()), axis=2)[mask]
        metadata = metadata.asnumpy()[mask]
        # update stats
        test_meter.update_stats(preds, ori_boxes, metadata)
        test_meter.log_iter_stats(None, cur_iter)
        test_meter.iter_tic()
    test_meter.finalize_metrics()

if __name__ == "__main__":
    run_eval()
