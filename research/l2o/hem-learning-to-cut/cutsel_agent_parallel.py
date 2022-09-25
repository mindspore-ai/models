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
import mindspore
import pyscipopt as scip
from pyscipopt import SCIP_RESULT

from third_party.logger import logger
from utils import advanced_cut_feature_generator


class CutSelectAgent(scip.Cutsel):
    def __init__(
            self,
            scip_model,
            pointer_net,
            value_net,
            sel_cuts_percent,
            device,
            decode_type,
            mean_std,
            policy_type
    ):
        super().__init__()
        self.scip_model = scip_model
        self.policy = pointer_net
        self.value = value_net
        self.sel_cuts_percent = sel_cuts_percent
        self.device = device
        self.decode_type = decode_type
        self.policy_type = policy_type

        self.data = {}
        self.lp_info = {
            "lp_solution_value": [],
            "lp_solution_integer_var_value": []
        }
        self.mean_std = mean_std

    def _normalize(self, cuts_features):
        normalize_features = (cuts_features - self.mean_std.mean) / \
                             (self.mean_std.std + self.mean_std.epsilon)
        return normalize_features

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if self.policy_type == 'with_token':
            cuts_dict = self._cutselselect_with_token(
                cuts, forcedcuts, root, maxnselectedcuts)
        else:
            cuts_dict = self._cutselselect(
                cuts, forcedcuts, root, maxnselectedcuts)

        return cuts_dict

    def _cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        logger.log("cut selection policy without token!!!")
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        num_cuts = len(cuts)
        cur_lp_info = self._get_lp_info()
        for k in cur_lp_info:
            self.lp_info[k].append(cur_lp_info[k])
        if num_cuts <= 1:
            return {
                'cuts': cuts,
                'nselectedcuts': 1,
                'result': SCIP_RESULT.SUCCESS
            }
        sel_cuts_num = min(
            int(num_cuts * self.sel_cuts_percent), int(maxnselectedcuts))
        sel_cuts_num = max(sel_cuts_num, 2)
        cuts_features = advanced_cut_feature_generator(self.scip_model, cuts)
        if self.mean_std is not None:
            normalize_cut_features = self._normalize(cuts_features)
            input_cuts = mindspore.from_numpy(
                normalize_cut_features).to(self.device)
        else:
            input_cuts = mindspore.from_numpy(cuts_features).to(self.device)

        input_cuts = input_cuts.reshape(
            input_cuts.shape[0], 1, input_cuts.shape[1])
        with mindspore.no_grad():
            _, input_idxs = self.policy(
                input_cuts.float(), sel_cuts_num, self.decode_type)
        idxes = [input.cpu().detach().item() for input in input_idxs]
        assert len(set(idxes)) == len(idxes)
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(idxes))
        sorted_cuts = [cuts[idx] for idx in idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)
        if not self.data:
            self.data = {
                "state": cuts_features,
                "action": idxes,
                "sel_cuts_num": sel_cuts_num,
            }

        return {
            'cuts': sorted_cuts,
            'nselectedcuts': sel_cuts_num,
            'result': SCIP_RESULT.SUCCESS
        }

    def _cutselselect_with_token(self,
                                 cuts,
                                 forcedcuts,
                                 root,
                                 maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        logger.log("cut selection policy with token!!!")
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        logger.log(f"maxnselectcuts: {maxnselectedcuts}")
        num_cuts = len(cuts)
        cur_lp_info = self._get_lp_info()
        for k in cur_lp_info:
            self.lp_info[k].append(cur_lp_info[k])
        if num_cuts <= 1:
            return {
                'cuts': cuts,
                'nselectedcuts': 1,
                'result': SCIP_RESULT.SUCCESS
            }
        max_sel_cuts_num = len(cuts) + 1
        cuts_features = advanced_cut_feature_generator(self.scip_model, cuts)
        if self.mean_std is not None:
            normalize_cut_features = self._normalize(cuts_features)
            input_cuts = mindspore.from_numpy(
                normalize_cut_features).to(self.device)
        else:
            input_cuts = mindspore.from_numpy(cuts_features).to(self.device)

        input_cuts = input_cuts.reshape(
            input_cuts.shape[0], 1, input_cuts.shape[1])
        with mindspore.no_grad():
            _, input_idxs = self.policy(
                input_cuts.float(), max_sel_cuts_num, self.decode_type)

        idxes = [input.cpu().detach().item() for input in input_idxs]
        sel_cuts_num = len(idxes)
        if not self.data:
            self.data = {
                "state": cuts_features,
                "action": idxes,
                "sel_cuts_num": sel_cuts_num,
            }
        assert idxes[-1] == num_cuts
        true_idxes = idxes[:-1]
        assert len(set(true_idxes)) == len(true_idxes)
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(true_idxes))
        sorted_cuts = [cuts[idx] for idx in true_idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)

        return {
            'cuts': sorted_cuts,
            'nselectedcuts': sel_cuts_num - 1,
            'result': SCIP_RESULT.SUCCESS
        }

    def _get_lp_info(self):
        lp_info = {}
        lp_info['lp_solution_value'] = self.scip_model.getLPObjVal()
        cols = self.scip_model.getLPColsData()
        col_solution_value = [col.getPrimsol()
                              for col in cols if col.isIntegral()]
        lp_info['lp_solution_integer_var_value'] = [
            val for val in col_solution_value if val != 0.]

        return lp_info

    def get_data(self):
        return self.data

    def get_lp_info(self):
        return self.lp_info

    def free_problem(self):
        self.scip_model.freeProb()


class HierarchyCutSelectAgent(CutSelectAgent):
    def __init__(
            self,
            scip_model,
            pointer_net,
            cutsel_percent_policy,
            value_net,
            sel_cuts_percent,
            device,
            decode_type,
            mean_std,
            policy_type
    ):
        CutSelectAgent.__init__(
            self,
            scip_model,
            pointer_net,
            value_net,
            sel_cuts_percent,
            device,
            decode_type,
            mean_std,
            policy_type
        )
        self.cutsel_percent_policy = cutsel_percent_policy
        self.high_level_data = {}

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        num_cuts = len(cuts)
        cur_lp_info = self._get_lp_info()
        for k in cur_lp_info:
            self.lp_info[k].append(cur_lp_info[k])
        if num_cuts <= 1:
            return {
                'cuts': cuts,
                'nselectedcuts': 1,
                'result': SCIP_RESULT.SUCCESS
            }

        cuts_features = advanced_cut_feature_generator(
            self.scip_model, cuts)

        if self.mean_std is not None:
            normalize_cut_features = self._normalize(cuts_features)
            input_cuts = mindspore.from_numpy(
                normalize_cut_features).to(self.device)
        else:
            input_cuts = mindspore.from_numpy(cuts_features).to(self.device)
        input_cuts = input_cuts.reshape(
            input_cuts.shape[0], 1, input_cuts.shape[1])

        deterministic = False
        with mindspore.no_grad():
            if self.decode_type == 'greedy':
                deterministic = True
            raw_sel_cuts_percent = self.cutsel_percent_policy.action(
                input_cuts.float(), deterministic=deterministic)

        sel_cuts_percent = raw_sel_cuts_percent.item() * 0.5 + 0.5
        sel_cuts_num = min(int(num_cuts * sel_cuts_percent),
                           int(maxnselectedcuts))
        sel_cuts_num = max(sel_cuts_num, 2)
        with mindspore.no_grad():
            _, input_idxs = self.policy(
                input_cuts.float(), sel_cuts_num, self.decode_type)

        idxes = [input.cpu().detach().item() for input in input_idxs]
        assert len(set(idxes)) == len(idxes)
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(idxes))
        sorted_cuts = [cuts[idx] for idx in idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)
        if not self.data:
            self.data = {
                "state": cuts_features,
                "action": idxes,
                "sel_cuts_num": sel_cuts_num,
            }
        if not self.high_level_data:
            self.high_level_data = {
                "state": cuts_features,
                "action": raw_sel_cuts_percent.item()
            }

        return {
            'cuts': sorted_cuts,
            'nselectedcuts': sel_cuts_num,
            'result': SCIP_RESULT.SUCCESS
        }

    def get_high_level_data(self):
        return self.high_level_data

#         _ = env.step(cutsel_agent)
