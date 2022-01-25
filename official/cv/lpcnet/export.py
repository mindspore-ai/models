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

from argparse import ArgumentParser

import numpy as np
import mindspore
from mindspore import context, export, load_checkpoint

from src import lpcnet

NB_USED_FEATURES = 20

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_file', '-c', type=str, default='../ckpt/lpcnet-4_37721.ckpt',
                        help='path of checkpoint')
    parser.add_argument('--max_len', type=int, default=500, help='number of 10ms frames')
    parser.add_argument('--out_file', '-n', type=str, default='lpcnet', help='name of model')
    parser.add_argument('--file_format', '-f', type=str, default='MINDIR', help='format of model')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend'],
                        help='device where the code will be implemented')
    parser.add_argument('--device_id', type=int, default=0,
                        help='which device where the code will be implemented')
    args = parser.parse_args()

    with open('ascend310_infer/inc/maxlen.h', 'w') as f:
        f.write(f"#define MAXLEN {args.max_len}")

    # NOTE: fails without max_call_depth due to RNN
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        max_call_depth=30000, device_id=args.device_id)

    model = lpcnet.WithLossLPCNet()
    model.backbone.to_float(mindspore.float16)
    load_checkpoint(args.ckpt_file, net=model)
    model.set_train(False)

    enc = model.backbone.encoder
    dec = model.backbone.decoder

    enc_input_feat = mindspore.Tensor(np.zeros([1, args.max_len, NB_USED_FEATURES]),
                                      mindspore.float32)
    enc_input_pitch = mindspore.Tensor(np.zeros([1, args.max_len, 1]), mindspore.int32)
    export(enc, enc_input_feat, enc_input_pitch, file_name=args.out_file+'_enc',
           file_format=args.file_format)
    print("Encoder exported successfully")

    dec_input_pcm = mindspore.Tensor(np.zeros([1, 1, 3]), mindspore.int32)
    dec_input_cfeat = mindspore.Tensor(np.zeros([1, 1, 128]), mindspore.float16)
    dec_input_state1 = mindspore.Tensor(np.zeros([1, 1, model.rnn_units1]), mindspore.float32)
    dec_input_state2 = mindspore.Tensor(np.zeros([1, 1, model.rnn_units2]), mindspore.float32)

    export(dec, dec_input_pcm, dec_input_cfeat, dec_input_state1, dec_input_state2,
           file_name=args.out_file+'_dec', file_format=args.file_format)
    print("Decoder exported successfully")

    print("Total Parameters:", sum(p.asnumpy().size for p in model.trainable_params()))
