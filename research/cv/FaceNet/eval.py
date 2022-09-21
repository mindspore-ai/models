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

import argparse
import numpy as np

from mindspore import load_checkpoint, load_param_into_net
from mindspore.ops import stop_gradient
from mindspore.common import set_seed
from mindspore import context
from src.eval_metrics import evaluate
from src.LFWDataset import get_lfw_dataloader
from src.models import FaceNetModelwithLoss
set_seed(0)


parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument("--ckpt", type=str, default="")
parser.add_argument("--eval_root_dir", type=str, default="")
parser.add_argument("--eval_pairs_path", type=str, default="")
parser.add_argument("--eval_batch_size", type=int, default=64)

args = parser.parse_args()


def validate_lfw(model_eval, lfw_dataloader):
    distances, labels = [], []

    print("Validating on LFW! ...")
    for data in lfw_dataloader.create_dict_iterator():
        distance = model_eval.evaluate(data['img1'], data['img2'])
        label = data['issame']
        distance = stop_gradient(distance)
        label = stop_gradient(label)
        distances.append(distance.asnumpy())
        labels.append(label.asnumpy())

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    _, _, accuracy, _, _, _ = evaluate(distances, labels)
    print(np.mean(accuracy))


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=1)
    facenet = FaceNetModelwithLoss(num_classes=1001, margin=0.5, mode='eval', ckpt_path="")
    state_dict = load_checkpoint(ckpt_file_name=args.ckpt, net=facenet)
    print("Loading the trained models from ckpt")
    load_param_into_net(facenet, state_dict)
    lfwdataloader = get_lfw_dataloader(eval_root_dir=args.eval_root_dir,
                                       eval_pairs_path=args.eval_pairs_path,
                                       eval_batch_size=args.eval_batch_size)
    validate_lfw(facenet, lfwdataloader)
