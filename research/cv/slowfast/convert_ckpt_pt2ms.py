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
"""Convert checkpoint of pytorch to minspore."""
import torch
import mindspore as ms

def convert_key_pt2ms(pt_key):
    """Convert keys"""
    ms_key = 'net.' + pt_key
    ms_key = ms_key.replace('bn.weight', 'bn.bn2d.gamma')
    ms_key = ms_key.replace('bn.bias', 'bn.bn2d.beta')
    ms_key = ms_key.replace('bn.running_mean', 'bn.bn2d.moving_mean')
    ms_key = ms_key.replace('bn.running_var', 'bn.bn2d.moving_variance')
    return ms_key


def convert_ckpt_pt2ms(pt_ckpt_path, ms_ckpt_path=None):
    """Convert keys with params of path"""
    ckpt_pt = torch.load(pt_ckpt_path, map_location=torch.device('cpu'))
    model_state = ckpt_pt['model_state']
    model_state_mapped = {}
    for key, weight in model_state.items():
        if key.endswith('num_batches_tracked'):
            continue
        ms_key = convert_key_pt2ms(key)
        print(f"{key} {weight.numpy().shape} -> {ms_key}")
        model_state_mapped[ms_key] = weight.numpy()
    if ms_ckpt_path is not None:
        # verify pt ckpt with ms ckpt
        ckpt_ms = ms.load_checkpoint(ms_ckpt_path)
        ckpt_ms_new = {}
        for key, weight in ckpt_ms.items():
            if key.startswith('accum') \
                    or key.startswith('stat') \
                    or key in ['learning_rate', 'momentum']:
                continue
            assert key in model_state_mapped
            assert weight.shape == model_state_mapped[key].shape
        assert len(model_state_mapped) == len(ckpt_ms_new)

    ckpt_ms_ret = []
    for key, weight in model_state_mapped.items():
        param_dict = {}
        param_dict['name'] = key
        param_dict['data'] = ms.Tensor(weight)
        ckpt_ms_ret.append(param_dict)
    ms.save_checkpoint(ckpt_ms_ret, pt_ckpt_path + '.ckpt')
    print('converting finished')


if __name__ == '__main__':
    convert_ckpt_pt2ms(pt_ckpt_path='checkpoint_epoch_00020_best248.pyth',
                       ms_ckpt_path=None)
