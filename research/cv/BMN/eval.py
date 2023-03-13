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
"""eval"""

import logging
import matplotlib
import numpy as np
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, SummaryCollector
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.bmn import BMN, BMNWithEvalCell
from src.loss import get_mask
from src.config import config as cfg
from src.dataset import createDataset
from src.metrics import BMNMetric
from src.eval_proposal import ANETproposal
matplotlib.use('Agg')

set_seed(1)

logging.basicConfig()
logger = logging.getLogger(__name__)

logger.info("Training configuration:\n\v%s\n\v", (cfg.__str__()))

logger.setLevel(cfg.log_level)


def run_evaluation(ground_truth_filename, proposal_filename,
                   max_avg_nr_proposals=100,
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True)
    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    auc_ = anet_proposal.auc

    return (average_nr_proposals, average_recall, recall, auc_)


def plot_metric(cfg_, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    import matplotlib.pyplot as plt

    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx, :], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.),
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou=0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.),
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(cfg_.postprocessing.save_fig_path)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.platform)

    #datasets
    eval_dataset, tem_train_dict = createDataset(cfg, mode='eval')
    batch_num = eval_dataset.get_dataset_size()

    #network
    network = BMN(cfg.model)
    logger.info("Network created")

    #BMN specific
    bm_mask = get_mask(cfg.model.temporal_scale)

    #checkpoint
    param_dict = load_checkpoint(cfg.eval.checkpoint)
    load_param_into_net(network, param_dict)

    # train net
    eval_net = BMNWithEvalCell(network)

    # metrics
    metric = BMNMetric(cfg, subset="validation")

    #models
    model = Model(eval_net,
                  eval_network=eval_net,
                  metrics={"bmn_metric": metric},
                  loss_fn=None)

    #callbacks
    time_cb = TimeMonitor(data_size=batch_num)
    summary_collector = SummaryCollector(summary_dir=cfg.summary_save_dir, collect_freq=1)

    cbs = [time_cb]

    model.eval(valid_dataset=eval_dataset,
               callbacks=cbs,
               dataset_sink_mode=False)

    logger.info("Exp. %s - BMN collecting eval results success", (cfg.experiment_name))

    logger.info("Evaluatiom started")

    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid, auc = run_evaluation(
        cfg.data.gt,
        cfg.postprocessing.result_file,
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset='validation')

    plot_metric(cfg, uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)
    print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
    print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
    print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
    print("AR@100 is \t", np.mean(uniform_recall_valid[:, -1]))
    print("Area Under the AR vs AN curve: \t", auc)
