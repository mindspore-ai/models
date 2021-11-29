# Copyright 2021 Huawei Technologies Co., Ltd
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
"""This is callback program"""
import collections
import os
import six
from src.albert_for_finetune import AlbertSquad
from src.dataset import create_squad_dataset
from src.model_utils.config import albert_net_cfg
import mindspore.common.dtype as mstype
from mindspore import Model, Tensor, log as logger
from mindspore.train.callback import Callback
from mindspore.common import set_seed
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net

if six.PY2:
    import six.moves.cPickle as pickle
else:
    import pickle


class albert_callback(Callback):
    """Squad task callback"""
    def __init__(self, net, args_opt, steps_per_epoch, save_checkpoint_path):
        self.net = net
        self.best_f1 = 0.0
        self.best_exact_match = 0
        self.best_epoch = 0
        self.args_opt = args_opt
        self.steps_per_epoch = steps_per_epoch
        self.path_url = self.args_opt.output_path
        self.save_checkpoint_path = save_checkpoint_path

    def epoch_end(self, run_context):
        """epoch end"""

        set_seed(323)

        from src import tokenization
        from src.squad_utils import read_squad_examples, convert_examples_to_features
        from src.squad_get_predictions import get_result
        from src.squad_postprocess import SQuad_postprocess
        tokenizer = tokenization.FullTokenizer(vocab_file=self.args_opt.vocab_file_path, do_lower_case=True,
                                               spm_model_file=self.args_opt.spm_model_file)
        eval_examples = read_squad_examples(self.args_opt.eval_json_path, False)
        if self.args_opt.enable_modelarts:
            self.args_opt.predict_feature_left_file = os.path.join(self.args_opt.data_path,
                                                                   self.args_opt.predict_feature_left_file)

        if not os.path.exists(self.args_opt.predict_feature_left_file):
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=albert_net_cfg.seq_length,
                doc_stride=128,
                max_query_length=64,
                is_training=False,
                output_fn=None,
                do_lower_case=True)

            with open(self.args_opt.predict_feature_left_file, "wb") as fout:
                pickle.dump(eval_features, fout)
        else:
            with open(self.args_opt.predict_feature_left_file, "rb") as fin:
                eval_features = pickle.load(fin)
        ds = create_squad_dataset(batch_size=self.args_opt.eval_batch_size, repeat_count=1,
                                  data_file_path=eval_features,
                                  schema_file_path=self.args_opt.schema_file_path, is_training=False,
                                  do_shuffle=(self.args_opt.eval_data_shuffle.lower() == "true"))

        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        logger.info("cur_epoch: ", cur_epoch)

        finetune_checkpoint_root = self.args_opt.load_finetune_checkpoint_path
        finetune_checkpoint_path = os.path.join(self.save_checkpoint_path, 'squad-' + str(cur_epoch) + '_' + str(
            self.steps_per_epoch) + '.ckpt')

        print("finetune_checkpoint_path: ", finetune_checkpoint_path)
        outputs = do_eval(ds, finetune_checkpoint_path, self.args_opt.eval_batch_size)
        all_predictions, _ = get_result(outputs, eval_examples, eval_features)
        re_json = SQuad_postprocess(self.args_opt.eval_json_path, all_predictions, output_metrics="output.json")

        exact_match = re_json['exact_match']

        f1 = re_json['f1']
        if self.args_opt.device_num == 1 or (self.args_opt.device_id == 0 and self.args_opt.device_num == 8):
            if exact_match > self.best_exact_match or f1 > self.best_f1:
                if not self.args_opt.enable_modelarts:
                    self.path_url = finetune_checkpoint_root
                self.best_exact_match = exact_match
                self.best_f1 = f1
                self.best_epoch = cur_epoch
                save_checkpoint(self.net, os.path.join(self.path_url, 'best_f1_%.5f_match_%.5f.ckpt'
                                                       % (self.best_f1, self.best_exact_match)))
                if self.args_opt.enable_modelarts:
                    import moxing as mox
                    mox.file.copy_parallel(src_url=self.path_url,
                                           dst_url=self.args_opt.train_url)
                log_text = 'EPOCH: %d, f1: %.1f, exact_match: %0.1f' % (cur_epoch, f1, exact_match)
                logger.info(log_text)
                log_text = 'BEST f1: %0.1f, BEST exact_match: %0.1f, BEST EPOCH: %s' \
                           % (self.best_f1, self.best_exact_match, self.best_epoch)
                logger.info(log_text)


def do_eval(dataset=None, load_checkpoint_path="", eval_batch_size=1):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net = AlbertSquad(albert_net_cfg, False, 2)
    net.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net, param_dict)
    model = Model(net)
    output = []
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_log_prob", "end_log_prob"])
    columns_list = ["input_ids", "input_mask", "segment_ids", "unique_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, segment_ids, unique_ids = input_data
        start_positions = Tensor([1], mstype.float32)
        end_positions = Tensor([1], mstype.float32)
        is_impossible = Tensor([1], mstype.float32)
        logits = model.predict(input_ids, input_mask, segment_ids, start_positions,
                               end_positions, unique_ids, is_impossible)
        ids = logits[0].asnumpy()
        start = logits[1].asnumpy()
        end = logits[2].asnumpy()

        for i in range(eval_batch_size):
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_log_prob=start_logits,
                end_log_prob=end_logits))
    return output
