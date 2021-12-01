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
import argparse
import json
import os
from os.path import join

import numpy as np
import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model_mindspore.pretrain_ms import UniterThreeForPretrainingForTd

from src.data import data_column, create_dataset, get_batch_data
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM
from src.tools.logger import LOGGER
from src.tools import parse_with_config

bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']

bad_endings += ['the']

project_root = os.path.abspath(os.path.dirname(
    os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val


def main(opts):
    context.set_context(mode=context.PYNATIVE_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=0)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    device_num = 1
    rank = 0
    opts.rank = rank

    ds, metaloader = create_dataset(opts, device_num=device_num, rank=rank, column_name=data_column,
                                    token_size=opts.train_batch_size, full_batch=opts.full_batch, is_train=False)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)
    net_without_loss = UniterThreeForPretrainingForTd(opts.model_config, img_dim=opts.img_dim,
                                                      img_label_dim=IMG_LABEL_DIM,
                                                      audio_dim=opts.audio_dim, audio_label_dim=AUDIO_LABEL_DIM,
                                                      use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                      full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                      args=opts)

    ckpt_file = ""
    if ckpt_file == "":
        params_dict = None
    else:
        params_dict = load_checkpoint(ckpt_file)

    model = Model(net_without_loss)

    if params_dict:
        net_not_load = load_param_into_net(net_without_loss, params_dict)
        print("===============net_not_load================", net_not_load)

    validate_td(model, ds, opts, metaloader, task="tdThree")


def validate_td(model, val_ds, opts, metaloader, task):
    """ Validation for td task"""
    LOGGER.info("start running Text Decoder validation...")
    vocab = json.load(open(opts.vocab_path))
    predictions = []
    split = ''
    cap_idx = 0
    batch_idx = 0
    for batch in val_ds.create_dict_iterator():
        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
         audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks,
         taskId) = get_batch_data(batch)

        seq = model.predict(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                            audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                            txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                            mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                            txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                            txt_masks, img_token_gts, img_token_masks,
                            taskId)
        print("one iter:", batch_idx)
        print(seq)
        print(seq.shape)
        batch_idx += 1
        seq = seq.asnumpy()
        seq = np.squeeze(seq, axis=1)

        sents = decode_sequence(vocab, seq, split=split)

        ids = metaloader.return_ids()
        for _, sent in enumerate(sents):
            key = ids[cap_idx]
            cap_idx += 1

            entry = {'image_id': key, 'caption': sent}
            LOGGER.info("image_id: %s caption: %s", key, sent)
            predictions.append(entry)

    print(cap_idx)
    print(len(predictions))
    cap_path = join(opts.output_dir, 'cap')
    if not os.path.exists(cap_path):
        os.mkdir(cap_path)
    cap_name = "step_" + task + ".json"
    full_path = join(cap_path, cap_name)
    json.dump(predictions, open(full_path, "w"))

    results = {}

    if 'aic' in opts.path_eval:
        evaler = AICEvaler(opts.anno_file, opts.path_eval)
        results = evaler.eval(full_path, cut=opts.cut)
    elif "coco" in opts.path_eval:
        evaler = COCOEvaler(opts.anno_file, opts.path_eval)
        results = evaler.eval(full_path)

    val_log.update(results)


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.

def decode_sequence(ix_to_word, seq, split=' '):
    """ Decode sequence """
    N = seq.shape[0]
    D = seq.shape[1]
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + split
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j - 1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words) + flag])
        out.append(txt.replace(' ##', ''))
    return out


def str2bool(b):
    assert b.lower() in ["false", "true"]
    if b.lower() in ["false"]:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default="pretrain_three_modal_txt_img_audio_config.json",
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
    parser.add_argument("--load_ckpt", default=False, type=bool, help="")
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

    parser.add_argument("--audio_model_config", type=str,
                        help="path to audio generate model structure config json")
    parser.add_argument("--audio_preprocess_config", type=str)

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
