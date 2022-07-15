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
"""score complexes"""
import os
import numpy as np
import pandas as pd
import h5py
from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore.dataset as ds
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.net import SBNetWork
from src.logger import get_logger
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

from src.data import Featurizer, make_grid


def modelarts_pre_process():
    pass


def input_file(in_paths):
    "Check if input file exists"
    in_paths = os.path.abspath(in_paths)
    if not os.path.exists(in_paths):
        raise IOError('File %s does not exist.' % in_paths)
    return in_paths


def eval_dataset(configs, data_path):
    """eval dataset process"""
    featurizer = Featurizer()
    charge_column = featurizer.FEATURE_NAMES.index('partialcharge')
    coords = []
    features = []
    names = []

    with h5py.File(data_path, 'r') as f:
        for name in f:
            names.append(name)
            dataset = f[name]
            coords.append(dataset[:, :3])
            features.append(dataset[:, 3:])
    if configs.verbose:
        if configs.batch_size == 0:
            configs.logger.info('Predict for all complexes at once')
        else:
            configs.logger.info('%s samples per batch' % configs.batch_size)
    evaldata = []
    for crd, f in zip(coords, features):
        evaldata.append(make_grid(crd, f, max_dist=configs.max_dist,
                                  grid_resolution=configs.grid_spacing))
    batch_grid = np.vstack(evaldata)
    batch_grid[..., charge_column] /= configs.charge_scaler
    batch_grid = np.transpose(batch_grid, axes=(0, 4, 1, 2, 3))
    batch_grid = np.expand_dims(batch_grid, axis=1)
    print("batch grid: ", batch_grid.shape, names)
    return batch_grid, names


class EvalDatasetIter:
    """Evaluation dataset iterator"""

    def __init__(self, grids):
        self.grids = grids

    def __getitem__(self, index):
        return self.grids[index]

    def __len__(self):
        return len(self.grids)


def load_evaldata(configs, data_path):
    """dataset loader"""
    batch_grid, names = eval_dataset(configs, data_path)
    eval_data = EvalDatasetIter(batch_grid)
    eval_loader = ds.GeneratorDataset(eval_data, column_names=['grid'])
    type_cast_op = C.TypeCast(mstype.float32)
    eval_loader = eval_loader.map(type_cast_op, input_columns=['grid'])
    eval_loader = eval_loader.batch(batch_size=20)
    return eval_loader, names


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=5)
    network = SBNetWork(in_chanel=[19, 64, 128],
                        out_chanle=config.conv_channels,
                        dense_size=config.dense_sizes,
                        lmbda=config.lmbda,
                        isize=config.isize, keep_prob=1.0, is_training=False)
    network.set_train(False)
    hdf_file_path = input_file(config.hdf_file)
    data_loader, names = load_evaldata(config, hdf_file_path)
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)
    prediction = []
    for data in data_loader.create_dict_iterator(output_numpy=True):
        grids = data['grid']
        if len(grids.shape) == 5:
            grids = Tensor(grids)
        elif len(grids.shape) == 6:
            grids = Tensor(np.squeeze(grids, 1))
        else:
            config.logger.info("Wrong input shape, please check dataset preprocess.")
        preds = network(grids)
        prediction.append(preds.asnumpy())
    config.logger.info("Finishing Evaluate.......")
    results = pd.DataFrame({'name': names, 'prediction': np.vstack(prediction).flatten()})
    results.to_csv(config.pre_output, index=False)
    config.logger.info('Result saved to %s', config.pre_output)


if __name__ == '__main__':
    config.logger = get_logger('./', config.device_id)
    config.logger.save_args(config)
    path = config.predict_input
    paths = input_file(path)
    config.hdf_file = paths
    run_eval()
