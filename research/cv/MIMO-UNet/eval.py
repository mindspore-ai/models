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
eval MIMO_UNet
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from mindspore import context
from mindspore import dataset as ds
from mindspore.common import set_seed
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.mimo_unet import MIMOUNet
from src.config import config
from src.data_load import create_dataset_generator
from src.loss import ContentLoss
from src.metric import PSNR


def run_eval(args):
    """eval"""
    context.set_context(mode=context.GRAPH_MODE)

    random.seed(1)
    set_seed(1)
    np.random.seed(1)

    eval_dataset_generator = create_dataset_generator(Path(args.dataset_root, 'test'))
    eval_dataset = ds.GeneratorDataset(eval_dataset_generator, ["image", "label"],
                                       shuffle=False, num_parallel_workers=args.num_worker)

    eval_dataset = eval_dataset.batch(batch_size=args.eval_batch_size, drop_remainder=True)
    net = MIMOUNet()
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    content_loss = ContentLoss()
    model = Model(net, loss_fn=content_loss, metrics={"PSNR": PSNR()})
    print("eval...")
    results = model.eval(eval_dataset, dataset_sink_mode=False)
    print(results)
    if args.img_save_directory:
        print("saving images...")
        Path(args.img_save_directory).mkdir(parents=True, exist_ok=True)
        ds_iter = eval_dataset.create_tuple_iterator()
        for num, (image, _) in enumerate(ds_iter):
            pred = net(image)[2]
            pred = pred.clip(0, 1)
            pred += 0.5 / 255
            pred = pred.asnumpy()
            pred = (pred.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)

            im_pred = Image.fromarray(pred, 'RGB')
            im_pred.save(Path(args.img_save_directory, f"{num}_pred.png"))

            image = image.asnumpy()
            image = (image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
            image = Image.fromarray(image, 'RGB')
            image.save(Path(args.img_save_directory, f"{num}_blur.png"))

    return results


if __name__ == '__main__':
    run_eval(config)
