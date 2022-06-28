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
"""do eval"""
import os
import operator
import argparse
from mindspore import context, Model
from mindspore.nn.optim import AdamWeightDecay
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_group_size
from src.tokenization import CscTokenizer
from src.soft_masked_bert import SoftMaskedBertCLS
from src.finetune_config import optimizer_cfg, bert_cfg
from src.utils import compute_corrector_prf
from tqdm import tqdm

def do_eval(dataset, network, profile=None, trained_ckpt_path="", epoch_num=1):
    network.set_train(False)
    para_dict = load_checkpoint(trained_ckpt_path)
    load_param_into_net(network, para_dict)
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        params = network.trainable_params()
        optimizer = AdamWeightDecay(params)
    model = Model(network, optimizer=optimizer, amp_level="O3")
    results = []
    cor_acc_labels = []
    det_acc_labels = []
    results = []
    det_acc_labels = []
    cor_acc_labels = []
    print("come in prediction...")
    for _, data in tqdm(enumerate(dataset.create_dict_iterator())):
        original_tokens, cor_y, cor_y_hat, det_y_hat, det_labels, batch_seq_len = model.predict(data['wrong_ids'], \
        data['original_tokens'], data['original_tokens_mask'], data['correct_tokens'], data['correct_tokens_mask'], \
        data['original_token_type_ids'], data['correct_token_type_ids'])
        for src, tgt, predict, det_predict, det_label, seq_len in \
        zip(original_tokens, cor_y, cor_y_hat, det_y_hat, det_labels, batch_seq_len):
            seq_len_ = int((seq_len[0] - 2).asnumpy().tolist())
            _src = src[1: seq_len_ + 1].asnumpy().tolist()
            _tgt = tgt[1: seq_len_ + 1].asnumpy().tolist()
            _predict = predict[1: seq_len_ + 1].asnumpy().tolist()
            _det_predict = det_predict[1:seq_len_ + 1].asnumpy().tolist()
            _det_label = det_label[1:seq_len_ + 1].asnumpy().tolist()
            cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
            det_acc_labels.append(1 if operator.eq(_det_predict, _det_label) else 0)
            results.append((_src, _tgt, _predict))
    compute_corrector_prf(results)

def run_csc():
    """run csc task"""
    parser = argparse.ArgumentParser(description="run csc")
    parser.add_argument("--bert_ckpt", type=str, default="bert_base.ckpt")
    parser.add_argument("--device_target", type=str, default="Ascend")
    parser.add_argument("--name", type=str, default="SoftMaskedBertModel")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--hyper_params", type=float, default=0.8)
    parser.add_argument("--eval_dataset", type=str, default="./datasets/csc/dev.json")
    parser.add_argument("--baselr", type=float, default=0.00001)
    parser.add_argument("--bias_lr_factor", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=5e-8)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=bert_cfg.seq_length) #512
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--train_url", type=str, default="./datasets/csc")
    parser.add_argument("--data_url", type=str, default="./datasets/csc/dev.json")  # direction of the training dataset, such as s3://open-data/attachment/
    parser.add_argument("--enable_modelarts", type=bool, default=False)
    parser.add_argument("--pynative", type=bool, default=False)
    args_opt = parser.parse_args()
    local_ckpt_dir = './weight/' + args_opt.bert_ckpt

    if args_opt.enable_modelarts:
        import moxing as mox
        # show paths
        print("data_url")
        print(args_opt.data_url)
        print("train_url")
        print(args_opt.train_url)
        if mox.file.exists('/cache/dataset'):
            ret = mox.file.list_directory('/cache/dataset', recursive=True)
            print('/cache/dataseet: (recursive)')
            print(ret)
        cloud_data_url = args_opt.data_url
        local_root_dir = '/home/work/user-job-dir'
        local_data_dir = os.path.join(local_root_dir, "data")
        local_dev_file_dir = os.path.join(local_data_dir, "SoftMask_test", "dev.json")
        local_ckpt_dir = os.path.join(local_data_dir, "SoftMask_test", args_opt.bert_ckpt)
        local_vocab_dir = os.path.join(local_data_dir, "SoftMask_test", "bert-base-chinese-vocab.txt")
        local_model_dir = os.path.join(local_root_dir, "model")
        if mox.file.exists(local_data_dir) is False:
            mox.file.make_dirs(local_data_dir)
        if mox.file.exists(local_model_dir) is False:
            mox.file.make_dirs(local_model_dir)
        mox.file.copy_parallel(cloud_data_url, local_data_dir)
        print(local_data_dir + ":")
        ret = mox.file.list_directory(local_data_dir, recursive=True)
        print(ret)
        ckpt_name = args_opt.ckpt_dir.split('/')[-1]
        trained_ckpt_path = os.path.join(local_data_dir, "SoftMask_test", ckpt_name)
    else:
        local_ckpt_dir = './weight/' + args_opt.bert_ckpt
        local_model_dir = './checkpoint'
        local_data_dir = args_opt.data_url
        local_dev_file_dir = local_data_dir
        trained_ckpt_path = args_opt.ckpt_dir
    # context setting
    if args_opt.device_target == "Ascend":
        if args_opt.enable_modelarts:
            rank_id = int(os.environ["DEVICE_ID"])
            if args_opt.pynative:
                context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=rank_id)
                args_opt.batch_size = 16
            else:
                context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=rank_id)
            init()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              gradients_mean=True,
                                              parallel_mode=context.ParallelMode.DATA_PARALLEL)
        else:
            device_id = args_opt.device_id
            device_num = 1
            if args_opt.pynative:
                context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=device_id)
                args_opt.batch_size = 16
            else:
                context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    else:
        raise Exception("Only support on Ascend currently.")
    netwithloss = SoftMaskedBertCLS(args_opt.batch_size, is_training=False, load_checkpoint_path=local_ckpt_dir)
    netwithloss.set_train(False)
    if args_opt.enable_modelarts:
        tokenizer = CscTokenizer(fp=local_dev_file_dir, device_num=device_num, rank_id=rank_id, \
        max_seq_len=args_opt.max_seq_len, vocab_path=local_vocab_dir)
    else:
        tokenizer = CscTokenizer(fp=local_dev_file_dir, device_num=device_num, rank_id=device_id, \
        max_seq_len=args_opt.max_seq_len)
    ds_eval = tokenizer.get_token_ids(args_opt.batch_size)
    do_eval(ds_eval, netwithloss, trained_ckpt_path=trained_ckpt_path)

if __name__ == "__main__":
    run_csc()
