# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This is the boot file for ModelArts platform.
Firstly, the train datasets are copied from obs to ModelArts.
Then, the string of train shell command is concated and using 'os.system()' to execute
"""
import os
import argparse
import time
import random
import datetime
import yaml
import numpy as np
import moxing as mox
import mindspore
from mindspore.context import ParallelMode
from mindspore.communication import init
from mindspore import Model, context, nn, DynamicLossScaleManager, Tensor, load_checkpoint, load_param_into_net, export
from src.metric import Sad
from src.dataset import create_dataset
from src.model import network
from src.loss import LossTNet, LossMNet, LossNet
from src.load_model import load_pre_model
from src.callback import TrainCallBack, EvalCallBack, LossMonitorSub
from src.config import get_config_from_yaml, update_config



print(os.system('env'))

def obs_data2modelarts(FLAGS):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(FLAGS.data_url_obs, FLAGS.data_url))
    mox.file.make_dirs(FLAGS.data_url)
    mox.file.copy_parallel(src_url=FLAGS.data_url_obs, dst_url=FLAGS.data_url)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(FLAGS.data_url)
    print("===>>>Files:", files)


def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=os.path.join(FLAGS.train_url), dst_url=FLAGS.train_url_obs)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.train_url,
                                                                                  FLAGS.train_url_obs))
    files = os.listdir()
    print("===>>>current Files:", files)
    mox.file.copy(src_url='shm_export.air', dst_url=FLAGS.train_url_obs+'/semantic_hm_best.air')


def export_AIR(args_opt):
    """
    Export Air model.
    """
    yaml_file = open(args_opt.yaml_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()

    y = yaml.load(file_data, Loader=yaml.FullLoader)
    cfg = y['export']
    cfg['ckpt_file'] = '/cache/output/single/ckpt_s2/end_to_end/semantic_hm_best.ckpt'
    cfg['file_format'] = 'AIR'
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])

    net = network.net(stage=2)
    param_dict = load_checkpoint(cfg['ckpt_file'])
    load_param_into_net(net, param_dict)
    net.set_train(False)

    x = Tensor(np.random.uniform(-1.0, 1.0, [1, 3, 320, 320]).astype(np.float32))
    export(net, x, file_name=cfg['file_name'], file_format=cfg['file_format'])


def init_env(cfg):
    """Init distribute env."""
    if cfg['saveIRFlag']:
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'], save_graphs=True,
                            save_graphs_path=os.path.join(cfg['saveIRGraph'], cur_time))
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'],
                            reserve_class_name_in_scope=False)

    device_num = int(os.getenv('RANK_SIZE', '1'))
    cfg['group_size'] = device_num
    print(f'device_num:{device_num}')

    if cfg['device_target'] == "Ascend":
        devid = int(os.getenv('DEVICE_ID', '0'))
        cfg['rank'] = devid
        print(f'device_id:{devid}')
        context.set_context(device_id=devid)
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True, parameter_broadcast=False)
    else:
        raise ValueError("Unsupported platform.")



class CustomWithLossCell(nn.Cell):
    """
    Train network wrapper
    """

    def __init__(self, backbone, loss_fn_t, loss_fn_m, loss_fn, stage=0):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn_t = loss_fn_t
        self._loss_fn_m = loss_fn_m
        self._loss_fn = loss_fn
        self._stage = stage

    def construct(self, img, trimap_ch_gt, trimap_gt, alpha_gt):
        if self._stage == 0:
            trimap_pre = self._backbone(img)
            return self._loss_fn_t(trimap_pre, trimap_gt)
        if self._stage == 1:
            alpha_pre = self._backbone(img, trimap_ch_gt)
            return self._loss_fn_m(img, alpha_pre, alpha_gt)
        trimap_pre, alpha_pre = self._backbone(img)
        return self._loss_fn(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt)


class WithEvalCell(nn.Cell):
    """
    Evaluation network wrapper
    """

    def __init__(self, net, loss_fn_t_net, loss_fn_m_net, loss_fn, stage=0):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._backbone = net
        self._loss_fn_t_net = loss_fn_t_net
        self._loss_fn_m_net = loss_fn_m_net
        self._loss_fn = loss_fn
        self._stage = stage

    def construct(self, img, trimap_ch_gt, trimap_gt, alpha_gt):
        if self._stage == 0:
            trimap_pre = self._backbone(img)
            return trimap_pre, trimap_gt, self._loss_fn_t_net(trimap_pre, trimap_gt)
        if self._stage == 1:
            alpha_pre = self._backbone(img, trimap_ch_gt)
            return alpha_pre, alpha_gt, self._loss_fn_m_net(img, alpha_pre, alpha_gt)
        trimap_pre, alpha_pre = self._backbone(img)
        return alpha_pre, alpha_gt, self._loss_fn(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt)


def run_train(cfg):
    """
    Start train process
    """
    dataset_train, _ = create_dataset(cfg, 'train', 1)
    dataset_eval, _ = create_dataset(cfg, 'eval', 1)

    dict_stage = {'pre_train_t_net': 0, 'pre_train_m_net': 1, 'end_to_end': 2}
    net = network.net(stage=dict_stage[cfg['train_phase']])
    cur_epoch = load_pre_model(net, cfg)
    print('----> total epoch: {}, current epoch: {}'.format(str(cfg['nEpochs']), str(cur_epoch)))
    if cfg['nEpochs'] <= cur_epoch:
        return

    net_loss = CustomWithLossCell(net, LossTNet(), LossMNet(), LossNet(), dict_stage[cfg['train_phase']])
    net_eval = WithEvalCell(net, LossTNet(), LossMNet(), LossNet(), dict_stage[cfg['train_phase']])

    scale_factor = 4
    scale_window = 3000
    loss_scale_manager = DynamicLossScaleManager(scale_factor, scale_window)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=float(cfg['lr']), weight_decay=0.0005)
    model = Model(network=net_loss, optimizer=optim, metrics={'sad': Sad()},
                  eval_network=net_eval, amp_level="O0", loss_scale_manager=loss_scale_manager)

    if cfg['rank'] == 0:
        print('----> rank 0 is training.')
        call_back = TrainCallBack(cfg=cfg, network=net, model=model, eval_callback=[EvalCallBack()],
                                  eval_dataset=dataset_eval, cur_epoch=cur_epoch, per_print_times=1)
    else:
        print('----> rank {} is training.'.format(str(cfg['rank'])))
        call_back = LossMonitorSub(cur_epoch=cur_epoch)
    model.train(epoch=cfg['nEpochs'] - cur_epoch, train_dataset=dataset_train,
                callbacks=[call_back], dataset_sink_mode=False)


def update_config_from_args(cfg, arg):
    """
    update some setting of yaml from args.
    """
    if arg.nEpochs_t != -1:
        cfg['pre_train_t_net']['nEpochs'] = arg.nEpochs_t
    if arg.nEpochs_m != -1:
        cfg['pre_train_m_net']['nEpochs'] = arg.nEpochs_m
    if arg.nEpochs_e != -1:
        cfg['end_to_end']['nEpochs'] = arg.nEpochs_e


if __name__ == '__main__':
    ## Note: the code dir is not the same as work dir on ModelArts Platform!!!
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    os.system('pip install -r {}'.format(os.path.join(code_dir, "requirements.txt")))
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser(description='semantic human matting !')
    parser.add_argument("--train_url_obs", type=str, default="./output")
    parser.add_argument("--data_url_obs", type=str, default="./dataset")
    parser.add_argument("--data_url", type=str, default="/cache/datasets")
    parser.add_argument("--train_url", type=str, default="/cache/output")#modelarts train result: /cache/output

    parser.add_argument("--nEpochs_t", type=int, default=-1, help="num of T net epochs")
    parser.add_argument("--nEpochs_m", type=int, default=-1, help="num of M epochs")
    parser.add_argument("--nEpochs_e", type=int, default=-1, help="num of end2end epochs")
    parser.add_argument('--yaml_path', type=str, default="config.yaml", help='config path')
    parser.add_argument('--init_weight', type=str, default="init_weight.ckpt", help='init weight path, optional')
    args = parser.parse_args()
    print(args)
    args.yaml_path = os.path.join(code_dir, args.yaml_path)
    args.init_weight = os.path.join(code_dir, args.init_weight)
    config = get_config_from_yaml(args)
    ## copy dataset from obs to modelarts
    obs_data2modelarts(args)
    random.seed(config['seed'])
    mindspore.set_seed(config['seed'])
    mindspore.common.set_seed(config['seed'])
    init_env(config)
    update_config(config)
    update_config_from_args(config, args)
    print("------------------------------config------------------------------")
    print(config)
    print("------------------------------------------------------------------")
    # Perform multistage train, the M-Net train phase is optional
    run_train(config['pre_train_t_net'])  # T-Net train phase
    # run_train(config['pre_train_m_net'])  # M-Net train phase
    run_train(config['end_to_end'])  # End-to-End train phase
    ## start export air
    export_AIR(args)
    ## copy result from modelarts to obs
    modelarts_result2obs(args)
