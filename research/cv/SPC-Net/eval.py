# Copyright 2023 Huawei Technologies Co., Ltd
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

import warnings
import time
import argparse
from network.network import deep_r50v3plusd
from src.dataset import create_dataset
from src.utils import save_imgs, fast_hist, evaluate_eval_for_inference

import mindspore as ms
from mindspore import context
import mindspore.ops as ops

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--root', type=str, default='/path/to/Datasets')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        help='[cityscapes, bdd, mapillary, synthia, gtav]')
    parser.add_argument('--num', type=int, default=2,
                        help='the number of sources. 1, 2 or 3')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    # define device
    ms.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
    # init network
    net = deep_r50v3plusd(args=args, num_classes=19, criterion=None, criterion_aux=None)
    # load paramwters
    if args.num == 1:
        MODELNAME = './models/single_source_model.ckpt'
    elif args.num == 2:
        MODELNAME = './models/double_source_model.ckpt'
    elif args.num == 3:
        MODELNAME = './models/triple_source_model.ckpt'
    else:
        raise AttributeError('No this mode!')
    param_dict = ms.load_checkpoint(MODELNAME)
    param_not_load = ms.load_param_into_net(net, param_dict)
    assert not param_not_load

    # init dataset
    dataset = create_dataset(root=args.root, data=args.dataset)

    # inference
    net.set_train(False)
    dataset_size = dataset.get_dataset_size()
    print('Start inference...')
    print('eval dataset size: {}'.format(dataset_size))
    IOUACC = 0
    start = time.time()
    for idx, data in enumerate(dataset.create_dict_iterator(num_epochs=1), start=0):
        image = data['image']
        label = data['label']
        img_name = data['img_name']
        prediction = net(image)
        prediction, _ = ops.ArgMaxWithValue(axis=1)(prediction)
        prediction = ops.Squeeze(0)(prediction)
        label = ops.Squeeze(0)(label)
        prediction = prediction.asnumpy()
        label = label.asnumpy()

        if args.save_path is not None:
            save_imgs(prediction=prediction, img_name=img_name, save_path=args.save_path)

        temp_iou = fast_hist(prediction.flatten(), label.flatten(), 19)

        IOUACC += temp_iou
    print('COST TIME:', (time.time()-start))
    res = evaluate_eval_for_inference(IOUACC, dataset_name=args.dataset)
    try:
        acc, acc_cls, mean_iu, fwavacc = res["acc"], res["acc_cls"], res["mean_iu"], res["fwavacc"]
        print('acc=%.6f, acc_cls=%.6f, mean_iu=%.6f, fwavacc=%.6f' % (acc, acc_cls, mean_iu, fwavacc))
    except KeyError:
        print("res format error, some key not found in res dict.")
