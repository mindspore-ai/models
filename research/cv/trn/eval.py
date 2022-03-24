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
"""Model evaluation"""

from mindspore import Model
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import nn
from mindspore import set_seed
from mindspore.nn.metrics import Top1CategoricalAccuracy
from mindspore.nn.metrics import Top5CategoricalAccuracy
from mindspore.train.callback import Callback
from mindspore.train.callback import TimeMonitor

from model_utils.config import config
from model_utils.logging import get_logger
from src.bn_inception import BNInception
from src.trn import RelationModuleMultiScale
from src.tsn import TSN
from src.tsn_dataset import get_dataset_for_evaluation

set_seed(config.seed)


def initialize_trn_network(trn_net, checkpoint_path):
    """Initialize network with the specified checkpoint"""
    ckpt_data = load_checkpoint(checkpoint_path)
    not_loaded = load_param_into_net(trn_net, ckpt_data)
    if not_loaded:
        print(f'The following parameters are not loaded: {not_loaded}')


class EvalWrapper(nn.Cell):
    """Wrapper for the model evaluation"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x, combinations, label):
        bs = x.shape[0]

        net_out = self.net(x, combinations)
        chs = net_out.shape[-1]

        net_out = net_out.reshape(bs, -1, chs)
        avg_output = net_out.mean(1)
        return avg_output, label


class MetricsMonitor(Callback):
    """Auxiliary class for Monitoring the metrics during the evaluation"""

    def __init__(self, metrics, print_steps_interval=1, logger=None):
        super().__init__()
        self._logger = logger
        self._metrics = metrics
        self._print_steps_interval = print_steps_interval

    def _log_metrics(self, step):
        metrics_data = {name: metric_object.eval() for name, metric_object in self._metrics.items()}
        metrics_str = '; '.join(f'{name}: {value:.6f}' for name, value in metrics_data.items())
        log_string = f'Steps: {step}; {metrics_str}'

        if self._logger is not None:
            self._logger.info(log_string)
        else:
            print(log_string, flush=True)

    def step_end(self, run_context):
        """Step end"""
        if self._print_steps_interval <= 0:
            return

        cb_params = run_context.original_args()

        if cb_params.cur_step_num <= 1:
            # Avoid trying to calculate the metrics before it is updated for the first time
            return

        if cb_params.cur_step_num % self._print_steps_interval == 0:
            self._log_metrics(cb_params.cur_step_num)


def run_eval(cfg):
    """Run evaluation"""
    logger = get_logger(cfg.eval_output_dir, 0)
    logger.save_args(cfg)

    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)

    dataset, num_class = get_dataset_for_evaluation(
        dataset_root=cfg.dataset_root,
        images_dir_name=cfg.images_dir_name,
        files_list_name=cfg.eval_list_file_name,
        image_size=cfg.image_size,
        num_segments=cfg.num_segments,
        subsample_num=cfg.subsample_num,
        seed=cfg.seed,
    )

    logger.important_info('start create network')

    backbone = BNInception(out_channels=cfg.img_feature_dim, dropout=cfg.dropout)
    trn_head = RelationModuleMultiScale(
        cfg.img_feature_dim,
        cfg.num_segments,
        num_class,
        subsample_num=cfg.subsample_num,
    )
    network = TSN(
        base_network=backbone,
        consensus_network=trn_head,
    )

    logger.info('load_checkpoint: %s', cfg.ckpt_file)
    initialize_trn_network(network, cfg.ckpt_file)
    logger.info('Checkpoint loaded!')

    logger.important_info('Validation')

    metrics = {'top1': Top1CategoricalAccuracy(), 'top5': Top5CategoricalAccuracy()}
    metrics_monitor = MetricsMonitor(metrics, print_steps_interval=100, logger=logger)

    model = Model(
        network,
        eval_network=EvalWrapper(network),
        metrics=metrics,
    )
    metrics = model.eval(dataset, callbacks=[TimeMonitor(), metrics_monitor], dataset_sink_mode=False)

    logger.info('Top1: {:.2f}%'.format(metrics['top1'] * 100))
    logger.info('Top5: {:.2f}%'.format(metrics['top5'] * 100))


if __name__ == '__main__':
    run_eval(config)
