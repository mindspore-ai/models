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
import json
import mindspore
from mindspore import context
from mindspore.train.model import Model
from src.model_mindspore.pretrain_ms import UniterThreeForPretrainingForVQAFinetune
from src.data import data_column, create_dataset, get_batch_data_vqa
from src.tools.logger import LOGGER
from src.tools import parse_with_config
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)



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

    ans2label = json.load(open(f'{dirname(abspath(__file__))}'
                               f'/utils/ans2label.json'))
    label2ans = {label: ans for ans, label in ans2label.items()}

    ds, _ = create_dataset(opts, device_num=device_num, rank=rank, column_name=data_column,
                           token_size=opts.train_batch_size, full_batch=opts.full_batch, is_train=False)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)
    #
    #
    #
    net_without_loss = UniterThreeForPretrainingForVQAFinetune(opts.model_config, img_dim=opts.img_dim,
                                                               img_label_dim=IMG_LABEL_DIM,
                                                               audio_dim=opts.audio_dim,
                                                               audio_label_dim=AUDIO_LABEL_DIM,
                                                               use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                               full_batch=opts.full_batch, use_moe=opts.use_moe)
    net_without_loss.init_output()

    model = Model(net_without_loss)

    results, logits = evaluate(model, ds, label2ans)
    print(results, logits)

def validate(model, val_ds, label2ans):
    """validate vqa"""
    LOGGER.info("start running validation...")

    results = []
    for batch in val_ds.create_dict_iterator():
        qids = batch['qids']
        (input_ids, position_ids, img_feat,
         img_pos_feat, attention_mask, gather_index, targets) = get_batch_data_vqa(batch)
        scores = model.predict(input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
                               gather_index, targets, compute_loss=False)

        # bs = batch['input_ids'].shape[0]

        answers = [label2ans[i]
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        for qid, answer in zip(qids, answers):
            results.append({'answer': answer, 'question_id': int(qid)})

    return results

def compute_score_with_logits(logits, labels):
    """compute score with logits"""
    logits = mindspore.ops.ArgMaxWithValue(logits, 1)[1]  # argmax
    one_hots = mindspore.ops.Zeros(*labels.shape)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


if __name__ == "__main__":
    default_path = "/home/work/user-job-dir/uniter-three/config/pretrain_three_modal_txt_img_audio_config.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=default_path, help='JSON config files')

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
    parser.add_argument("--load_ckpt", default=True, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps", default=5000, type=int, help="")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of data.')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
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
