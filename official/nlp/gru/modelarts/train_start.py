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
"""train script"""
import os
import subprocess
import time

import numpy as np
from mindspore import context
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Adam
from mindspore.train import Model
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from model_utils.config import config
from model_utils.device_adapter import get_rank_id, get_device_id, get_device_num
from src.dataset import create_gru_dataset
from src.gru_for_infer import GRUInferCell
from src.gru_for_train import GRUWithLossCell, GRUTrainOneStepWithLossScaleCell
from src.lr_schedule import dynamic_lr
from src.seq2seq import Seq2Seq

set_seed(1)


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))


time_stamp_init = False
time_stamp_first = 0


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')


def run_export(ckpt_file_name, out_file_name, file_format='AIR'):
    """run export."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, reserve_class_name_in_scope=False,
                        device_id=get_device_id(), save_graphs=False)
    network = Seq2Seq(config, is_training=False)
    network = GRUInferCell(network)
    network.set_train(False)
    if config.ckpt_file != "":
        parameter_dict = load_checkpoint(ckpt_file_name)
        load_param_into_net(network, parameter_dict)

    source_ids = Tensor(np.random.uniform(0.0, 1e5, size=[config.eval_batch_size, config.max_length]).astype(np.int32))
    target_ids = Tensor(np.random.uniform(0.0, 1e5, size=[config.eval_batch_size, config.max_length]).astype(np.int32))
    export(network, source_ids, target_ids, file_name=out_file_name, file_format=file_format)


def _create_tokenized_sentences(input_path, output_path, file, language):
    """
    Create tokenized sentences files.

    Args:
        input_path: input path
        output_path: output path
        file: file name.
        language: text language
    """
    from nltk.tokenize import word_tokenize
    sentence = []
    total_lines = open(os.path.join(input_path, file), "r").read().splitlines()
    for line in total_lines:
        line = line.strip('\r\n ')
        line = line.lower()
        tokenize_sentence = word_tokenize(line, language)
        str_sentence = " ".join(tokenize_sentence)
        sentence.append(str_sentence)
    tokenize_file = os.path.join(output_path, file + ".tok")
    f = open(tokenize_file, "w")
    for line in sentence:
        f.write(line)
        f.write("\n")
    f.close()


def _merge_text(input_path, output_path, file_list, output_file):
    """
    Merge text files together.

    Args:
        input_path: input path
        output_path: output path
        file_list: dataset files list.
        output_file: output file after merge
    """
    output_file = os.path.join(output_path, output_file)
    f_output = open(output_file, "w")
    for file_name in file_list:
        text_path = os.path.join(input_path, file_name) + ".tok"
        f = open(text_path)
        f_output.write(f.read() + "\n")
    f_output.close()


def data_preprocess():
    """prepare data for traning"""
    from src.preprocess import get_dataset_vocab
    config.mr_data_save_path = os.path.join(config.train_url, 'mr_data')
    config.processed_data_save_path = os.path.join(config.train_url, 'data')
    if os.path.exists(os.path.join(config.mr_data_save_path, config.dataset_path)):
        return
    if not os.path.exists(config.processed_data_save_path):
        os.makedirs(config.processed_data_save_path)
    if not os.path.exists(config.mr_data_save_path):
        os.makedirs(config.mr_data_save_path)

    # install tokenizer
    # nltk.download('punkt')
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        zip_isexist = zipfile.is_zipfile(zip_file)
        if zip_isexist:
            fz = zipfile.ZipFile(zip_file, 'r')
            data_num = len(fz.namelist())
            print("Extract Start...")
            print("unzip file num: {}".format(data_num))
            data_print = int(data_num / 100) if data_num > 100 else 1
            i = 0
            for file in fz.namelist():
                if i % data_print == 0:
                    print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                i += 1
                fz.extract(file, save_dir)
            print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                 int(int(time.time() - s_time) % 60)))
            print("Extract Done.")
        else:
            print("This is not zip.")

    if not os.path.exists(os.path.join(os.environ['HOME'], 'nltk_data')):
        unzip_path = os.path.join(os.environ['HOME'], 'nltk_data', 'tokenizers')
        os.makedirs(unzip_path)
        unzip(os.path.join(config.data_url, 'punkt.zip'), unzip_path)

    # preprocess
    src_file_list = ["train.de", "test.de", "val.de"]
    dst_file_list = ["train.en", "test.en", "val.en"]
    for file in src_file_list:
        _create_tokenized_sentences(config.data_url, config.processed_data_save_path, file, "english")
    for file in dst_file_list:
        _create_tokenized_sentences(config.data_url, config.processed_data_save_path, file, "german")
    src_all_file = "all.de.tok"
    dst_all_file = "all.en.tok"
    _merge_text(config.processed_data_save_path, config.processed_data_save_path, src_file_list, src_all_file)
    _merge_text(config.processed_data_save_path, config.processed_data_save_path, dst_file_list, dst_all_file)
    src_vocab = os.path.join(config.processed_data_save_path, "vocab.de")
    dst_vocab = os.path.join(config.processed_data_save_path, "vocab.en")
    get_dataset_vocab(os.path.join(config.processed_data_save_path, src_all_file), src_vocab)
    get_dataset_vocab(os.path.join(config.processed_data_save_path, dst_all_file), dst_vocab)

    # paste
    cmd = f'paste {config.processed_data_save_path}/train.de.tok \
        {config.processed_data_save_path}/train.en.tok > \
        {config.processed_data_save_path}/train.all'
    os.system(cmd)
    cmd = f'paste {config.processed_data_save_path}/test.de.tok \
        {config.processed_data_save_path}/test.en.tok > \
        {config.processed_data_save_path}/test.all'
    os.system(cmd)

    # create data
    create_data_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "src/create_data.py")
    cmd = ["python", create_data_file, '--num_splits=8',
           f"--input_file={config.processed_data_save_path}/train.all",
           f"--src_vocab_file={config.processed_data_save_path}/vocab.de",
           f"--trg_vocab_file={config.processed_data_save_path}/vocab.en",
           f"--output_file={config.mr_data_save_path}/multi30k_train_mindrecord",
           '--max_seq_length=32', '--bucket=[32]']
    print(f"Start preprocess, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()
    cmd = ["python", create_data_file, '--num_splits=1',
           f"--input_file={config.processed_data_save_path}/test.all",
           f"--src_vocab_file={config.processed_data_save_path}/vocab.de",
           f"--trg_vocab_file={config.processed_data_save_path}/vocab.en",
           f"--output_file={config.mr_data_save_path}/multi30k_test_mindrecord",
           '--max_seq_length=32', '--bucket=[32]']
    print(f"Start preprocess, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def run_train():
    """run train."""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=get_device_id(), save_graphs=False)
    rank = get_rank_id()
    device_num = get_device_num()
    if config.run_distribute:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    mindrecord_file = os.path.join(config.mr_data_save_path, config.dataset_path)
    config.outputs_path = os.path.join(config.train_url, 'train')
    if not os.path.exists(config.outputs_path):
        os.makedirs(config.outputs_path)
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_gru_dataset(epoch_count=config.num_epochs, batch_size=config.batch_size,
                                 dataset_path=mindrecord_file, rank_size=device_num, rank_id=rank)
    dataset_size = dataset.get_dataset_size()
    print("dataset size is {}".format(dataset_size))
    network = Seq2Seq(config)
    network = GRUWithLossCell(network)
    lr = dynamic_lr(config, dataset_size)
    opt = Adam(network.trainable_params(), learning_rate=lr)
    scale_manager = DynamicLossScaleManager(init_loss_scale=config.init_loss_scale_value,
                                            scale_factor=config.scale_factor,
                                            scale_window=config.scale_window)
    update_cell = scale_manager.get_update_cell()
    netwithgrads = GRUTrainOneStepWithLossScaleCell(network, opt, update_cell)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    # Save Checkpoint
    if config.save_checkpoint:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_epoch * dataset_size,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.outputs_path, 'ckpt_' + str(get_rank_id()) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(get_rank_id()))
        cb += [ckpt_cb]
    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    model.train(config.num_epochs, dataset, callbacks=cb, dataset_sink_mode=True)
    run_export(os.path.join(save_ckpt_path, str(rank) + '-' + str(config.num_epochs) + '_1807.ckpt'),
               os.path.join(save_ckpt_path, 'gru'), 'AIR')
    run_export(os.path.join(save_ckpt_path, str(rank) + '-' + str(config.num_epochs) + '_1807.ckpt'),
               os.path.join(save_ckpt_path, 'gru'), 'MINDIR')


if __name__ == '__main__':
    data_preprocess()
    run_train()
