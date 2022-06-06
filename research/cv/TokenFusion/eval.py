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

"""Evaluate mIou and Pixacc"""
import os
import argparse
import numpy as np
import cv2
import mindspore.ops as ops
from mindspore import load_param_into_net
from mindspore import load_checkpoint
import mindspore.dataset as ds
from mindspore import context
import mindspore.common.dtype as mstype
from config import TRAIN_DIR, VAL_DIR, TRAIN_LIST, VAL_LIST, IGNORE_LABEL, SHORTER_SIDE, CROP_SIZE, RESIZE_SIZE, NORMALISE_PARAMS, NUM_CLASSES
from models.segformer import WeTr
from utils import confusion_matrix, make_validation_img, getScores, print_log
from utils.datasets import SegDataset
from utils.transforms import (
    CropAlignToMask,
    ResizeAlignToMask,
    ResizeInputs,
    Normalise,
    ToBatchTensor,
)

parser = argparse.ArgumentParser(description="ICNet Evaluation")
parser.add_argument(
    "--dataset_path", type=str, default="/data/cityscapes/", help="dataset path"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="tokenfusion_mitb3_nyudv2.ckpt",
    help="checkpoint_path, default67.7",
)
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
# Dataset
parser.add_argument(
    "-d",
    "--train-dir",
    type=str,
    default=TRAIN_DIR,
    help="Path to the training set directory.",
)
parser.add_argument(
    "--val-dir", type=str, default=VAL_DIR, help="Path to the validation set directory."
)
parser.add_argument(
    "--train-list", type=str, default=TRAIN_LIST, help="Path to the training set list."
)
parser.add_argument(
    "--val-list", type=str, default=VAL_LIST, help="Path to the validation set list."
)
parser.add_argument(
    "--ignore-label",
    type=int,
    default=IGNORE_LABEL,
    help="Label to ignore during training",
)
parser.add_argument(
    "--shorter-side",
    type=int,
    default=SHORTER_SIDE,
    help="Shorter side transformation.",
)
parser.add_argument(
    "--crop-size", type=int, default=CROP_SIZE, help="Crop size for training,"
)
parser.add_argument(
    "--input-size", type=int, default=RESIZE_SIZE, help="Final input size of the model"
)
parser.add_argument(
    "--normalise-params",
    type=list,
    default=NORMALISE_PARAMS,
    help="Normalisation parameters [scale, mean, std],",
)
parser.add_argument(
    "-i",
    "--input_types",
    default=["rgb", "depth"],
    type=str,
    nargs="+",
    help="input type (image, depth)",
)
parser.add_argument(
    "--batch-size", type=int, default=1, help="Batch size to train the segmenter model."
)
parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                    help='Number of output classes for each task.')
args_opt = parser.parse_args()

num_classes = args_opt.num_classes


class Evaluator:
    """evaluate"""

    def __init__(self):
        # create network
        self.model = WeTr("mit_b3", num_classes, pretrained=False)

        # create dataloader
        dataset = "nyudv2"
        AlignToMask = CropAlignToMask if dataset == "nyudv2" else ResizeAlignToMask
        print(args_opt.input_size)
        composed_val = [
            AlignToMask(),
            ResizeInputs(args_opt.input_size),
            Normalise(*args_opt.normalise_params),
        ]
        input_names, input_mask_idxs = ["rgb", "depth"], [0, 2, 1]
        self.validset = SegDataset(
            dataset=dataset,
            data_file=args_opt.val_list,
            data_dir=args_opt.val_dir,
            input_names=input_names,
            input_mask_idxs=input_mask_idxs,
            transform_trn=None,
            transform_val=composed_val,
            stage="val",
            ignore_label=args_opt.ignore_label,
        )
        self.val_loader = ds.GeneratorDataset(
            self.validset,
            column_names=["rgb", "depth", "mask"],
            shuffle=False,
            num_parallel_workers=1,
        )
        self.val_loader = self.val_loader.batch(
            args_opt.batch_size, drop_remainder=False
        )
        self.val_loader = self.val_loader.create_dict_iterator(output_numpy=False)
        # load ckpt
        ckpt_file_name = args_opt.checkpoint_path
        param_dict = load_checkpoint(ckpt_file_name)
        load_param_into_net(self.model, param_dict)

    def eval(self):
        save_image = 0
        segmenter = self.model
        segmenter = segmenter.set_train(False)

        segmenter.to_float(mstype.float32)

        input_types = args_opt.input_types
        conf_mat = []
        for _ in range(len(input_types) + 1):
            conf_mat.append(np.zeros((num_classes, num_classes), dtype=int))
        for i, sample in enumerate(self.val_loader):
            print("{}/{}".format(i, int(len(self.validset) / args_opt.batch_size)))
            sample = ToBatchTensor()(sample)
            rgb = sample["rgb"]
            depth = sample["depth"]
            target = sample["mask"]
            gt = target[0].asnumpy().astype(np.uint8)
            gt_idx = (
                gt < num_classes
            )  # Ignore every class index larger than the number of classes
            inputs = [rgb, depth]
            outputs, _ = segmenter(inputs)
            for idx, output in enumerate(outputs):
                output = (
                    cv2.resize(
                        output[0, :num_classes].asnumpy().transpose(1, 2, 0),
                        target.shape[1:][::-1],
                        interpolation=cv2.INTER_CUBIC,
                    )
                    .argmax(axis=2)
                    .astype(np.uint8)
                )
                # Compute IoU
                conf_mat[idx] += confusion_matrix(
                    gt[gt_idx], output[gt_idx], num_classes
                )
                if i < save_image or save_image == -1:
                    img = make_validation_img(
                        inputs[0].asnumpy(),
                        inputs[1].asnumpy(),
                        ops.ExpandDims()(sample["mask"], 0).asnumpy(),
                        output[np.newaxis, :],
                    )
                    os.makedirs("imgs", exist_ok=True)
                    cv2.imwrite("imgs/validate_%d.png" % i, img[:, :, ::-1])
                    print("imwrite at imgs/validate_%d.png" % i)

        for idx, input_type in enumerate(input_types + ["ens"]):
            glob, mean, iou = getScores(conf_mat[idx])
            best_iou_note = ""
            alpha = "        "
            input_type_str = "(%s)" % input_type
            print_log(
                "%-7s   glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f%s%s"
                % (input_type_str, glob, mean, iou, alpha, best_iou_note)
            )
        print_log("")
        return iou


if __name__ == "__main__":
    context.set_context(
        mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False
    )
    evaluator = Evaluator()
    evaluator.eval()
