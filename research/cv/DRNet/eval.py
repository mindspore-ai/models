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
"""
eval.
"""

import argparse
import mindspore
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.predictor import predictor
from src.classifier import resnet50 as classifier
from src.gumbelmodule import GumbleSoftmax
import numpy as np

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--predictor_checkpoint_path', type=str, default='./checkpoint/predictor_net.ckpt',
                    help='Checkpoint file path')
parser.add_argument('--classifier_checkpoint_path', type=str, default='./checkpoint/classifier_net.ckpt',
                    help='Checkpoint file path')
parser.add_argument('--data_url', type=str, default='/home/ma-user/work/cache/data/imagenet', help='Dataset path')
args_opt = parser.parse_args()


if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
    predictor_net = predictor([1, 1, 1, 1])
    classifier_net = classifier()
    dataset = create_dataset(dataset_path=args_opt.data_url, do_train=False, infer_910=False)
    test_loader = dataset.create_dict_iterator(output_numpy=True)

    if args_opt.predictor_checkpoint_path:
        predictor_param_dict = load_checkpoint(args_opt.predictor_checkpoint_path)
        load_param_into_net(predictor_net, predictor_param_dict)
    predictor_net.set_train(False)

    if args_opt.classifier_checkpoint_path:
        classifier_param_dict = load_checkpoint(args_opt.classifier_checkpoint_path)
        load_param_into_net(classifier_net, classifier_param_dict)
    classifier_net.set_train(False)
    gumbel_softmax = GumbleSoftmax()
    predictor_net.set_train(False)
    classifier_net.set_train(False)
    interpolate = mindspore.nn.ResizeBilinear()
    resize_ratio = [224, 168, 112]
    img_tot = 0
    top1_correct = 0
    for batch_idx, imgs in enumerate(test_loader):
        images, target = mindspore.Tensor(imgs['image']), mindspore.Tensor(imgs['label'])
        predictor_input = interpolate(images, size=(128, 128))
        predictor_ratio_score = predictor_net(predictor_input)
        predictor_ratio_score_gumbel = gumbel_softmax(predictor_ratio_score)
        output = 0
        for j, r in enumerate(resize_ratio):
            new_images = interpolate(images, size=(r, r))
            new_output = classifier_net(new_images, int(j))
            output += predictor_ratio_score_gumbel[:, j:j+1] * new_output
        output = output.asnumpy()
        top1_output = np.argmax(output, (-1))
        t1_correct = np.equal(top1_output, target).sum()
        top1_correct += t1_correct
        batch_size = target.shape[0]
        img_tot += batch_size
    acc1 = 100.0 * top1_correct / img_tot
    print("result:", acc1)
    