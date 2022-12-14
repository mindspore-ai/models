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

from src.data import load_data
from src.dialogXL import DialogXL, ERC_xlnet
from src.config import DialogXLConfig, init_args
from mindspore import export, load_checkpoint, load_param_into_net, context

if __name__ == '__main__':
    args = init_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device)
    config = DialogXLConfig()
    config.set_args(args)

    dialogXL = DialogXL(config)
    model = ERC_xlnet(args, dialogXL)
    param_dict = load_checkpoint('checkpoints/MELD_best.ckpt')
    load_param_into_net(model, param_dict)

    trainsets = load_data('data/MELD_trainsets.pkl')
    content_ids, _, content_mask, content_lengths, speaker_ids = trainsets[0][0]
    mems, speaker_mask, window_mask = None, None, None
    export(model, content_ids, mems, content_mask, content_lengths, speaker_ids, speaker_mask, window_mask,
           file_name='dialogXL', file_format='MINDIR')
    print('========================================')
    print('dialogXL.mindir exported successfully!')
    print('========================================')
