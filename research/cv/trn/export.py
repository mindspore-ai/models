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
"""Export models into different formats"""
from pathlib import Path

from mindspore import context
from mindspore import set_seed
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from src.bn_inception import BNInception
from src.trn import RelationModuleMultiScale
from src.tsn import TSN
from src.tsn_dataset import get_dataset_for_evaluation

set_seed(config.seed)


def modelarts_pre_process():
    """modelarts pre process function."""


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export(cfg):
    """Export model to the format specified by the user."""
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

    backbone = BNInception(out_channels=cfg.img_feature_dim, dropout=cfg.dropout)
    trn_head = RelationModuleMultiScale(
        cfg.img_feature_dim,
        cfg.num_segments,
        num_class,
        subsample_num=cfg.subsample_num,
    )
    net = TSN(
        base_network=backbone,
        consensus_network=trn_head,
    )

    load_checkpoint(cfg.ckpt_file, net=net)
    net.set_train(False)

    input_data = next(iter(dataset.create_tuple_iterator()))

    export_dir = Path(cfg.export_output_dir).resolve()
    export_file = export_dir / cfg.model_name
    print("export directory:", export_dir)
    print("model name:", cfg.model_name)
    print("export file format:", cfg.file_format)
    export(net, input_data[0], input_data[1], file_name=str(export_file), file_format=cfg.file_format)


if __name__ == '__main__':
    run_export(config)
