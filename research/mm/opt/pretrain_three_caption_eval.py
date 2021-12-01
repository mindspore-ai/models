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
from aic_caption.pycxevalcap.eval import COCOEvalCap
from aic_caption.pycxtools.coco import COCO
import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.data.generator import get_batch_data_captioneval
from src.model_mindspore.pretrain_ms import UniterThreeForPretrainingForCapfinetuneEval
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM
from src.tools.logger import LOGGER
from src.tools import parse_with_config

from pretrain_three_data import create_three_dataloaders


bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']
bad_endings += ['the']

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val


def main(opts):
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    local_rank = rank_id
    print('local_rank:{}, device id:{}'.format(local_rank, device_id))

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    device_num = 1
    rank = 0
    opts.rank = rank

    test_loaders = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, False,
                                            opts, device_num=device_num)
    test_loader = test_loaders['ftCap']

    net_without_loss = UniterThreeForPretrainingForCapfinetuneEval(opts.model_config, img_dim=opts.img_dim,
                                                                   img_label_dim=IMG_LABEL_DIM,
                                                                   audio_dim=opts.audio_dim,
                                                                   audio_label_dim=AUDIO_LABEL_DIM,
                                                                   use_txt_out=opts.use_txt_out,
                                                                   use_video=opts.use_video,
                                                                   full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                                   args=opts, batch_size=opts.val_batch_size)

    ckpt_file = opts.ckpt_file
    print(ckpt_file)
    if ckpt_file == "":
        modified_params_dict = None
    else:
        params_dict = load_checkpoint(ckpt_file)

        modified_params_dict = {}
        for k, v in params_dict.items():
            if 'txt_output.tfm_decoder' in k:
                modified_k = k.replace('txt_output.tfm_decoder', 'txt_output.tfm_decoder.decoder.tfm_decoder')
                v.name = v.name.replace('txt_output.tfm_decoder', 'txt_output.tfm_decoder.decoder.tfm_decoder')
                modified_v = v
                modified_params_dict[modified_k] = modified_v
            else:
                modified_params_dict[k] = v

    if modified_params_dict:
        net_not_load = load_param_into_net(net_without_loss, modified_params_dict)
        print("===============net_not_load================", net_not_load)

    validate_td(net_without_loss, test_loader, opts, task="caption", ckpt_file=ckpt_file)


def validate_td(model, test_loader, opts, task, ckpt_file):
    """
     validate_td
    """
    LOGGER.info("start running Text Decoder validation...")

    vocab = json.load(open(opts.vocab_path))

    cap_path = join(opts.output_dir, 'capeval')
    if not os.path.exists(cap_path):
        os.mkdir(cap_path)
    cap_name = ckpt_file.split('/')[-1][:-5] + ".json"
    full_path = join(cap_path, cap_name)

    if os.path.exists(full_path):
        eval_result = compute_metric(opts.caption_gt_eval, full_path)
        print(eval_result)

    else:
        predictions = []
        split = ' '
        cap_idx = 0
        batch_idx = 0
        total = 0
        for batch in test_loader:
            ids = batch['ids']
            (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
             audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
             txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
             mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
             txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
             txt_masks, img_token_gts, img_token_masks,
             taskId) = get_batch_data_captioneval(batch)

            seq = model(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                        audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                        txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                        mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                        txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                        txt_masks, img_token_gts, img_token_masks,
                        taskId)
            total += seq.shape[0]
            print("one iter:", batch_idx)
            print(seq)
            print(seq.shape)
            print("already_processed: ", total)

            seq = seq.asnumpy()
            seq = np.squeeze(seq, axis=1)
            seq = seq[:, 1:]
            sents = decode_sequence(vocab, seq, split=split)

            for k, sent in enumerate(sents):
                image_id = ids[k].split('.jpg')[0][-6:]
                entry = {'image_id': image_id, 'caption': sent}
                LOGGER.info("image_id:%d caption:%s", image_id, sent)
                predictions.append(entry)

        print(cap_idx)
        print(len(predictions))

        json.dump(predictions, open(full_path, "w"))

        eval_result = compute_metric(opts.caption_gt_eval, full_path)
        print(eval_result)


def process_gt_gile(gt_path, gt_processed_path):
    """
    process_gt_gile
    """
    src = json.load(open(gt_path))
    tgt = {}
    tgt['annotations'] = []
    for k, v in src.items():
        while len(k) < 6:
            k = '0' + k
        for vs in v:
            js = {'image_id': k, 'caption': vs, 'id': k}
            tgt['annotations'].append(js)
    print(len(tgt['annotations']))
    json.dump(tgt, open(gt_processed_path, 'w'))


def process_predict_gile(predict_path, predict_processed_path):
    src = json.load(open(predict_path))
    tgt = []
    for i in src:
        v = {}
        v['image_id'] = i['image_id']
        v['caption'] = ''.join(i['caption'].split(' '))
        tgt.append(v)
    print(len(tgt))
    json.dump(tgt, open(predict_processed_path, 'w'))


def compute_metric(gt_path, predict_path):
    """
    compute_metric
    """
    gt_processed_path = gt_path.split('.json')[-2] + '_processed' + '.json'
    predict_processed_path = predict_path.split('.json')[-2] + '_processed' + '.json'
    process_gt_gile(gt_path, gt_processed_path)

    if not os.path.exists(predict_processed_path):
        process_predict_gile(predict_path, predict_processed_path)

    coco = COCO(gt_processed_path)
    cocoRes = coco.loadRes(predict_processed_path, cut=False)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    return cocoEval.eval


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq, split=' '):
    """
    decode_sequence
    """
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
    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=1024, type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048, type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True, type=str2bool, help='use txt out')

    parser.add_argument('--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument('--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument('--name_audio', default="audio2len_three.json", type=str, help='use audio out')

    parser.add_argument("--init_loss_scale", default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument("--load_ckpt", default=False, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps", default=5000, type=int, help="")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of data.')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument('--sink_size', default=2, type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")
    parser.add_argument('--ckpt_file', default="", type=str, help='use txt out')
    parser.add_argument("--audio_model_config", type=str,
                        help="path to audio generate model structure config json")
    parser.add_argument("--audio_preprocess_config", type=str)
    parser.add_argument('--output_dir', default="", type=str, help='use audio out')
    parser.add_argument('--caption_gt_eval', default="/store0/dataset/coco_data/coco_trans_captions.json", type=str,
                        help='use audio out')

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
