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
vqa gen model
"""
import copy

import mindspore.nn as nn

from transformer_config import transformer_net_cfg as cfg
from model_mindspore.loss import TransformerTrainingLoss
from model_mindspore.model_config import UniterConfig
from model_mindspore.model_ms import UniterThreeModel
from model_mindspore.parallel_transformer import ParallelConfig
from model_mindspore.transformer_model import TransformerModel


class UniterThreeForPretrainingForVQAgenfinetune(nn.Cell):
    """ UNITER VQAGEN FINETUNE """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, args=None):
        super(UniterThreeForPretrainingForVQAgenfinetune, self).__init__()
        parallel_config = ParallelConfig()
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe)

        # Text Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        self.txt_cfg = copy.deepcopy(cfg)
        self.txt_output = TransformerModel(cfg, True, config.hidden_size, parallel_config, fg_backbone, False)
        self.td_crit = TransformerTrainingLoss(cfg, parallel_config)

    def generate_text(self, sequence_output, att_masks, txt_gts, txt_masks):
        txt_out = self.txt_output(sequence_output, att_masks, txt_gts, txt_masks)
        loss = self.td_crit(txt_out, txt_gts, txt_masks)
        return loss

    def construct(self, input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt,
                  img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target,
                  audio_mask_tgt, audio_masks, mafr_feat_target, itm_targets, ma_neg_index, ma_neg_sample,
                  mr_neg_index, mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks,
                  taskId):
        """UniterThreeForPretrainingForVQAgenfinetune construct"""
        sequence_output, _ = self.uniter(input_ids, position_ids,
                                         img_feat, img_pos_feat,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False,
                                         txt_type_ids=None, img_type_ids=None,
                                         audio_feat=None, audio_pos_ids=None,
                                         audio_type_ids=None)

        td_loss = self.generate_text(sequence_output, attention_mask, txt_gts, txt_masks)
        return td_loss
