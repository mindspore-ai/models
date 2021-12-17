# Copyright 2020 Huawei Technologies Co., Ltd
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
UNITER pre-training
"""
import os
import argparse
import math
import mindspore as ms
from mindspore import context
from mindspore.train.model import Model
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.ops as ops
from mindspore.communication.management import init, get_group_size, get_rank
from src.data import data_column, create_dataset, get_batch_data
from src.model_mindspore.pretrain_ms import UniterThreeForPretrainingForRet
from src.tools.logger import LOGGER
from src.tools import parse_with_config
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM
import numpy as np

project_root = os.path.abspath(os.path.dirname(
    os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def main(opts):
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="Ascend")
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    if opts.use_parallel:
        init()
        LOGGER.info("start init")

        device_num = get_group_size()
        rank = get_rank()
        opts.rank = rank
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num)
    else:
        device_num = 1
        rank = 0
        opts.rank = rank

    ds, _ = create_dataset(opts, device_num=device_num, rank=rank, column_name=data_column,
                           token_size=opts.train_batch_size, full_batch=opts.full_batch, is_train=False)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)

    net_without_loss = UniterThreeForPretrainingForRet(opts.model_config, img_dim=opts.img_dim,
                                                       img_label_dim=IMG_LABEL_DIM,
                                                       audio_dim=opts.audio_dim, audio_label_dim=AUDIO_LABEL_DIM,
                                                       use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                       full_batch=opts.full_batch, use_moe=opts.use_moe)

    ckpt_file = "/mnt/sfs_turbo/ckpt/OPT_11-5000_2.ckpt"
    #     ckpt_file = ""
    if ckpt_file == "":
        params_dict = None
    else:
        params_dict = load_checkpoint(ckpt_file)

    model = Model(net_without_loss)

    if params_dict:
        new_params_dict = {}
        for key in params_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_params_dict[key_new] = params_dict[key]
            else:
                new_params_dict[key] = params_dict[key]
        new_params_dict["uniter.img_embeddings.img_linear.weight"] = new_params_dict["feat_regress.weight"]
        new_params_dict["uniter.audio_embeddings.audio_linear.weight"] = new_params_dict["audio_feat_regress.weight"]
        new_params_dict["uniter.embeddings.word_embeddings.embedding_table"] = new_params_dict[
            "cls.predictions.decoder.weight"]
        net_not_load = load_param_into_net(net_without_loss, new_params_dict)
        print("===============net_not_load================", net_not_load)

    if params_dict:
        net_not_load = load_param_into_net(net_without_loss, params_dict)
        print("===============net_not_load================", net_not_load)

    validate_itm_matching(model, ds)


def validate_itm_matching(model, val_ds):
    """validate itm matching"""
    topk = ops.TopK()
    LOGGER.info("start running ITM validation...")
    score_vec = Tensor(np.zeros((1000000,)), ms.float32)
    k = 2

    n_ex = 0

    for batch in val_ds.create_dict_iterator():
        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
         audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks,
         taskId) = get_batch_data(batch)

        scores = model.predict(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                               audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                               txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                               mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                               txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                               txt_masks, img_token_gts, img_token_masks,
                               taskId)

        if get_rank() == 0:
            print("-------------------scores:", scores, "---------------")
            print(scores.shape)

    if get_rank() == 0:
        print("===========n_ex=", n_ex)
        score_vec = score_vec[:n_ex]
        print(score_vec.shape)
        print(score_vec)
        k = 10
        score_mat = score_vec.reshape((int(math.sqrt(n_ex)), -1))

        print(score_mat)
        print("........................", score_mat.dtype,
              score_mat.shape, int(math.sqrt(n_ex)))
        max_targets = np.arange(0, int(math.sqrt(n_ex)), dtype=np.int64)
        _, topk_indices = topk(score_mat, 10)
        topk_ind = topk_indices.asnumpy()
        gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
        _, rank = np.nonzero(topk_ind == gt_img_j)
        ir_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
        ir_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
        ir_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))
        print(ir_r1, ir_r5, ir_r10)

        score_mat = score_mat.T
        _, topk_indices = topk(score_mat, 10)
        topk_ind = topk_indices.asnumpy()
        gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
        _, rank = np.nonzero(topk_ind == gt_img_j)
        ir_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
        ir_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
        ir_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))
        print(ir_r1, ir_r5, ir_r10)


def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/home/work/user-job-dir/uniter-three_eval/config/test_ch.json",
                        help='JSON config files')
    parser.add_argument("--start_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=120000, type=int,
                        help="The decay step.")
    parser.add_argument('--use_txt_out', default=False,
                        type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False,
                        type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=True,
                        type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=1024,
                        type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048,
                        type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True,
                        type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True,
                        type=str2bool, help='use txt out')

    parser.add_argument(
        '--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument(
        '--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument(
        '--name_audio', default="audio2len_three.json", type=str, help='use audio out')

    parser.add_argument("--init_loss_scale",
                        default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument("--load_ckpt", default=True, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps",
                        default=5000, type=int, help="")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument('--data_url', required=True,
                        default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True,
                        default=None, help='Location of data.')
    parser.add_argument(
        "--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument('--sink_size', default=2, type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
