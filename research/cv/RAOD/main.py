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
# =======================================================================================
import os
import shutil
import argparse
import datetime

import numpy as np
from tqdm import tqdm
import mindspore as ms
import mindspore.context as context

from model import model
from model.util import postprocess
from dataset.dataset import create_dataset
from dataset.data_utils import DataPreprocess, DeMosaic, Resize, GrayWorldWB


current_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_path)


def main(arguments):
    # ## init
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    out_path = os.path.join(current_path, arguments.out_path, now)
    ckpt_path = os.path.join(current_path, arguments.ckpt_path)
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)

    if os.path.isfile(ckpt_path) and ckpt_path.endswith('.ckpt'):
        network = model()
        ms.load_checkpoint(ckpt_path, network, strict_load=True)
        network.set_train(False)

        # ## init dataset
        dataset = create_dataset(arguments, is_train=False)
        data_preproc = DataPreprocess(ops_=[DeMosaic, Resize, GrayWorldWB])

        # ## start inference
        print('\n\nStart inference ...')
        print(f'eval dataset size: {dataset.get_dataset_size()}')
        for batch in tqdm(dataset.create_dict_iterator(num_epochs=1)):
            raw, name = batch['raw'], batch['file_name'].asnumpy()

            rgb = data_preproc(raw)
            pred = network(rgb).asnumpy()
            output = postprocess(pred, num_classes=6, conf_thre=0.001, nms_thre=0.65)

            for n, out in zip(name, output):
                if out is None:
                    continue

                txt = os.path.join(out_path, n.replace('.raw', '.txt'))
                fd = os.open(txt, os.O_WRONLY | os.O_CREAT, mode=0o511)
                with os.fdopen(fd, 'w') as f:
                    out[:, :4] = np.clip(out[:, :4], 0, 1280)
                    for o in out:
                        x1, y1, x2, y2, conf1, conf2, klass = o.tolist()
                        if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                            f.write('%d %.6f %d %d %d %d\n' % (klass + 1, conf1*conf2, x1, y1, x2, y2))

    else:
        print(f'Invalid ckpt < {ckpt_path} >!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_path', type=str, default='./model.ckpt')
    parser.add_argument('-i', '--in_path', type=str, default='valid_list.txt')
    parser.add_argument('-o', '--out_path', type=str, default='../results')
    parser.add_argument('-d', '--device', type=str, default='GPU')
    parser.add_argument('--input_shape', type=int, default=[1856, 2880], nargs=2)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    main(args)
