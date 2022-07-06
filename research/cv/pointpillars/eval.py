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
"""Evaluation script"""

import argparse
import os
import warnings
from time import time

from mindspore import context, Tensor
from mindspore import dataset as de
from mindspore import load_checkpoint
from mindspore import load_param_into_net

from src.core.eval_utils import get_official_eval_result
from src.predict import predict
from src.predict import predict_kitti_to_anno
from src.utils import get_config
from src.utils import get_model_dataset
from src.utils import get_params_for_net

warnings.filterwarnings('ignore')


def run_evaluate(args):
    """run evaluate"""
    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path

    cfg = get_config(cfg_path)

    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_target = args.device_target

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id)

    model_cfg = cfg['model']

    center_limit_range = model_cfg['post_center_limit_range']

    pointpillarsnet, eval_dataset, box_coder = get_model_dataset(cfg, False)

    params = load_checkpoint(ckpt_path)
    new_params = get_params_for_net(params)
    load_param_into_net(pointpillarsnet, new_params)

    eval_input_cfg = cfg['eval_input_reader']

    eval_column_names = eval_dataset.data_keys

    ds = de.GeneratorDataset(
        eval_dataset,
        column_names=eval_column_names,
        python_multiprocessing=True,
        num_parallel_workers=6,
        max_rowsize=100,
        shuffle=False
    )
    batch_size = eval_input_cfg['batch_size']
    ds = ds.batch(batch_size, drop_remainder=False)
    data_loader = ds.create_dict_iterator(num_epochs=1, output_numpy=True)

    class_names = list(eval_input_cfg['class_names'])

    dt_annos = []
    gt_annos = [info["annos"] for info in eval_dataset.kitti_infos]

    log_freq = 100
    len_dataset = len(eval_dataset)
    start = time()
    for i, data in enumerate(data_loader):
        voxels = data["voxels"]
        num_points = data["num_points"]
        coors = data["coordinates"]
        bev_map = data.get('bev_map', False)

        preds = pointpillarsnet(Tensor(voxels), Tensor(num_points), Tensor(coors), Tensor(bev_map))
        if len(preds) == 2:
            preds = {
                'box_preds': preds[0].asnumpy(),
                'cls_preds': preds[1].asnumpy(),
            }
        else:
            preds = {
                'box_preds': preds[0].asnumpy(),
                'cls_preds': preds[1].asnumpy(),
                'dir_cls_preds': preds[2].asnumpy()
            }
        preds = predict(data, preds, model_cfg, box_coder)

        dt_annos += predict_kitti_to_anno(preds,
                                          data,
                                          class_names,
                                          center_limit_range)

        if i % log_freq == 0 and i > 0:
            time_used = time() - start
            print(f'processed: {i * batch_size}/{len_dataset} imgs, time elapsed: {time_used} s',
                  flush=True)

    result = get_official_eval_result(
        gt_annos,
        dt_annos,
        class_names,
    )
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', required=True, help='Path to config file.')
    parser.add_argument('--ckpt_path', required=True, help='Path to checkpoint.')
    parser.add_argument('--device_target', default='GPU', help='device target')
    parser.add_argument('--is_modelarts', default='0', help='')
    parser.add_argument('--data_url', default='', help='')
    parser.add_argument('--train_url', default='', help='')

    parse_args = parser.parse_args()
    if parse_args.is_modelarts == '1':
        import moxing as mox
        data_dir = '/home/work/user-job-dir/data'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        obs_data_url = parse_args.data_url
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    run_evaluate(parse_args)
