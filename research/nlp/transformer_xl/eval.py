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
from mindspore import load_checkpoint, context
from mindspore.dataset import GeneratorDataset
from src.callback.eval import doEval
from src.metric.calc import bpc
from src.model.mem_transformer import MemTransformerLM
from src.model.mem_transformer_for_ascend import MemTransformerLM as MemTransformerLMAscend
from src.model_utils.config import config
from src.utils.dataset_util import get_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer-XL evaluation running')
    parser.add_argument('--datadir', default='./data/enwik8',
                        help='Directory contains enwik8 dataset.')
    parser.add_argument('--dataset', default='enwik8',
                        help='Dataset Name.', choices=["enwik8", "text8"])
    parser.add_argument('--ckpt_path', default="./model0.ckpt", help='Directory of model.')
    parser.add_argument("--device_target", type=str, default="GPU", help="Device Target, default GPU",
                        choices=["Ascend", "GPU"])
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")

    args = parser.parse_args()
    datadir = args.datadir
    dataset = args.dataset
    device_id = args.device_id

    dataset = get_dataset(datadir, dataset)
    ntokens = len(dataset.vocab)

    context.set_context(device_id=device_id)
    if args.device_target == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, enable_graph_kernel=True)

        # Due to the mems mechanism, it is not possible to perform multi-card segmentation on the valid and test datasets
        valid_dataset = GeneratorDataset(source=dataset.get_valid_generator(), column_names=['data', 'target'],
                                         shuffle=False)
        test_dataset = GeneratorDataset(source=dataset.get_test_generator(), column_names=['data', 'target'],
                                        shuffle=False)

        # adaptive softmax / embedding
        cutoffs = []
        net = MemTransformerLM(ntokens, config.n_layer, config.n_head, config.d_model,
                               config.d_head, config.d_inner, config.dropout, config.dropatt,
                               batch_size=config.batch_size,
                               d_embed=config.d_embed, div_val=config.div_val,
                               pre_lnorm=config.pre_lnorm, tgt_len=config.tgt_len,
                               ext_len=config.ext_len, mem_len=config.mem_len, eval_tgt_len=config.eval_tgt_len,
                               cutoffs=cutoffs, same_length=config.same_length, clamp_len=config.clamp_len)

        # model_filename = os.path.join(config.load_path, args.ckpt_filename + '.ckpt')
        model_filename = args.ckpt_path
        print(model_filename)
        load_checkpoint(net=net, ckpt_file_name=model_filename)

        valid_loss = doEval(net, valid_dataset, config.tgt_len, config.ext_len, config.mem_len, config.eval_tgt_len)
        test_loss = doEval(net, test_dataset, config.tgt_len, config.ext_len, config.mem_len, config.eval_tgt_len)

        print('=' * 100)
        if config.dataset in ['enwik8', 'text8']:
            print('| End of valid | valid loss {:5.2f} | valid bpc {:9.5f}'.format(
                valid_loss, bpc(valid_loss)))
            print('| End of test | test loss {:5.2f} | test bpc {:9.5f}'.format(
                test_loss, bpc(test_loss)))
        print('=' * 100)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, enable_graph_kernel=False)

        # Due to the mems mechanism, it is not possible to perform multi-card segmentation on the valid and test datasets
        test_dataset = GeneratorDataset(source=dataset.get_test_generator(), column_names=['data', 'target'],
                                        shuffle=False)

        # adaptive softmax / embedding
        cutoffs = []
        net = MemTransformerLMAscend(ntokens, config.n_layer, config.n_head, config.d_model, config.d_head,
                                     config.d_inner, config.dropout, config.dropatt, batch_size=config.batch_size,
                                     d_embed=config.d_embed, div_val=config.div_val, pre_lnorm=config.pre_lnorm,
                                     tgt_len=config.tgt_len, ext_len=config.ext_len, mem_len=config.mem_len,
                                     eval_tgt_len=config.eval_tgt_len, cutoffs=cutoffs, same_length=config.same_length,
                                     clamp_len=config.clamp_len)

        # model_filename = os.path.join(config.load_path, args.ckpt_filename + '.ckpt')
        model_filename = args.ckpt_path
        print(model_filename)
        load_checkpoint(net=net, ckpt_file_name=model_filename,
                        filter_prefix=['mems', 'valid_mems', 'empty_valid_mems'])

        test_loss = doEval(net, test_dataset, config.tgt_len, config.ext_len, config.mem_len, config.eval_tgt_len)

        print('=' * 100)
        if config.dataset in ['enwik8', 'text8']:
            print('| End of test | test loss {:5.2f} | test bpc {:9.5f}'.format(
                test_loss, bpc(test_loss)))
        print('=' * 100)
