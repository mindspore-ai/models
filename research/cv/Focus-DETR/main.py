# Copyright 2023 Huawei Technologies Co., Ltd
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import mindspore
from mindspore import context

from models.focus_detr.coco_eval import CocoEvaluator, post_process
from models.focus_detr.dataset import build_dataset
from models.focus_detr.focus_detr import build_focus_detr
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    #
    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str, default="/comp_robot/cv_public_dataset/COCO2017/")
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")

    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--rank", default=0, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", type=int, help="local rank for DistributedDataParallel")
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser


class dataset_param():
    def __init__(self):
        self.eval = True
        self.max_img_size = 1333
        self.num_queries = 900
        self.img_scales = [480, 512, 640, 800]
        self.device_num = 1
        self.num_classes = 91
        self.rank = 0
        self.num_workers = 1
        self.batch_size = 1
        self.coco_path = "coco2017/"


def main(args):
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    # update some new args temporally
    if not getattr(args, "use_ema", None):
        args.use_ema = False
    if not getattr(args, "debug", None):
        args.debug = False
    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"), distributed_rank=args.rank, color=False, name="detr"
    )
    logger.info(f"Command: {' '.join(sys.argv)}")
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Full config saved to {save_json_path}")
    logger.info(f"world size: {args.world_size}")
    logger.info(f"rank: {args.rank}")
    logger.info(f"local_rank: {args.local_rank}")
    logger.info(f"args: {args}\n")
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)

    device_id = int(os.getenv("DEVICE_ID", "0"))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=device_id)
    focus_detr = build_focus_detr(args)

    param_dict = mindspore.load_checkpoint(args.resume)
    param_not_load = mindspore.load_param_into_net(focus_detr, param_dict)
    print(f"-----param_not_load:{param_not_load}")
    focus_detr.set_train(False)

    ds_param = dataset_param()
    dataset, base_ds, _ = build_dataset(ds_param)
    data_loader = dataset.create_dict_iterator()
    coco_evaluator = CocoEvaluator(base_ds, ["bbox"])

    cnt = 0
    test_cnt = 100

    for sample in data_loader:
        images = sample["image"]
        mask_ms = sample["mask"]
        input_data = {"data": images, "mask": mask_ms}
        outputs = focus_detr(input_data)
        outputs = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}

        orig_target_sizes = sample["orig_sizes"].asnumpy()
        results = post_process(outputs, orig_target_sizes)
        res = {img_id: output for img_id, output in zip(sample["img_id"].asnumpy(), results)}
        coco_evaluator.update(res)
        cnt += 1

        if cnt % 10 == 0:
            print(f"---process--img--nums:{cnt}")
        if cnt > test_cnt:
            break
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


if __name__ == "__main__":
    main_args = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()]).parse_args()
    if main_args.output_dir:
        Path(main_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(main_args)
