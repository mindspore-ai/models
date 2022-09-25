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
import os

import numpy as np
import pyscipopt as scip
from logger import logger


class SCIPCutSelEnv():
    def __init__(
            self,
            instance_file_path,
            scip_seed,
            seed,
            scip_time_limit=3600,
            single_instance_file=None,
            **init_scip_kwargs
    ):
        self.instance_file_path = instance_file_path
        self.instances = os.listdir(instance_file_path)
        self.single_instance_file = single_instance_file
        self.scip_seed = scip_seed
        self.seed = seed
        self.scip_time_limit = scip_time_limit
        self.init_scip_kwargs = init_scip_kwargs

        # self.reset()
        self.set_seed()

    def _set_scip_separator_params(self, max_rounds_root=-1, max_rounds=-1, max_cuts_root=10000, max_cuts=10000,
                                   frequency=10):
        """
        Function for setting the separator params in SCIP. It goes through all separators, enables them at all points
        in the solving process,
        Args:
            scip: The SCIP Model object
            max_rounds_root: The max number of separation rounds that can be performed at the root node
            max_rounds: The max number of separation rounds that can be performed at any non-root node
            max_cuts_root: The max number of cuts that can be added per round in the root node
            max_cuts: The max number of cuts that can be added per node at any non-root node
            frequency: The separators will be called each time the tree hits a new multiple of this depth
        Returns:
            The SCIP Model with all the appropriate parameters now set
        """

        assert isinstance(max_cuts, int) and isinstance(max_rounds, int)
        assert isinstance(max_cuts_root, int) and isinstance(max_rounds_root, int)

        model = self.m

        # First for the aggregation heuristic separator
        model.setParam('separating/aggregation/freq', frequency)
        model.setParam('separating/aggregation/maxrounds', max_rounds)
        model.setParam('separating/aggregation/maxroundsroot', max_rounds_root)
        model.setParam('separating/aggregation/maxsepacuts', max_cuts)
        model.setParam('separating/aggregation/maxsepacutsroot', max_cuts_root)

        # Now the Chvatal-Gomory w/ MIP separator
        # model.setParam('separating/cgmip/freq', frequency)
        # model.setParam('separating/cgmip/maxrounds', max_rounds)
        # model.setParam('separating/cgmip/maxroundsroot', max_rounds_root)

        # The clique separator
        model.setParam('separating/clique/freq', frequency)
        model.setParam('separating/clique/maxsepacuts', max_cuts)

        # The close-cuts separator
        model.setParam('separating/closecuts/freq', frequency)

        # The CMIR separator
        model.setParam('separating/cmir/freq', frequency)

        # The Convex Projection separator
        model.setParam('separating/convexproj/freq', frequency)
        model.setParam('separating/convexproj/maxdepth', -1)

        # The disjunctive cut separator
        model.setParam('separating/disjunctive/freq', frequency)
        model.setParam('separating/disjunctive/maxrounds', max_rounds)
        model.setParam('separating/disjunctive/maxroundsroot', max_rounds_root)
        model.setParam('separating/disjunctive/maxinvcuts', max_cuts)
        model.setParam('separating/disjunctive/maxinvcutsroot', max_cuts_root)
        model.setParam('separating/disjunctive/maxdepth', -1)

        # The separator for edge-concave function
        model.setParam('separating/eccuts/freq', frequency)
        model.setParam('separating/eccuts/maxrounds', max_rounds)
        model.setParam('separating/eccuts/maxroundsroot', max_rounds_root)
        model.setParam('separating/eccuts/maxsepacuts', max_cuts)
        model.setParam('separating/eccuts/maxsepacutsroot', max_cuts_root)
        model.setParam('separating/eccuts/maxdepth', -1)

        # The flow cover cut separator
        model.setParam('separating/flowcover/freq', frequency)

        # The gauge separator
        model.setParam('separating/gauge/freq', frequency)

        # Gomory MIR cuts
        model.setParam('separating/gomory/freq', frequency)
        model.setParam('separating/gomory/maxrounds', max_rounds)
        model.setParam('separating/gomory/maxroundsroot', max_rounds_root)
        model.setParam('separating/gomory/maxsepacuts', max_cuts)
        model.setParam('separating/gomory/maxsepacutsroot', max_cuts_root)

        # The implied bounds separator
        model.setParam('separating/impliedbounds/freq', frequency)

        # The integer objective value separator
        model.setParam('separating/intobj/freq', frequency)

        # The knapsack cover separator
        model.setParam('separating/knapsackcover/freq', frequency)

        # The multi-commodity-flow network cut separator
        model.setParam('separating/mcf/freq', frequency)
        model.setParam('separating/mcf/maxsepacuts', max_cuts)
        model.setParam('separating/mcf/maxsepacutsroot', max_cuts_root)

        # The odd cycle separator
        model.setParam('separating/oddcycle/freq', frequency)
        model.setParam('separating/oddcycle/maxrounds', max_rounds)
        model.setParam('separating/oddcycle/maxroundsroot', max_rounds_root)
        model.setParam('separating/oddcycle/maxsepacuts', max_cuts)
        model.setParam('separating/oddcycle/maxsepacutsroot', max_cuts_root)

        # The rapid learning separator
        model.setParam('separating/rapidlearning/freq', frequency)

        # The strong CG separator
        # model.setParam('separating/strongcg/freq', frequency)
        # model.setParam('separating/strongcg/maxrounds', max_rounds)
        # model.setParam('separating/strongcg/maxroundsroot', max_rounds_root)
        # model.setParam('separating/strongcg/maxsepacuts', max_cuts)
        # model.setParam('separating/strongcg/maxsepacutsroot', max_cuts_root)

        # The zero-half separator
        model.setParam('separating/zerohalf/freq', frequency)
        model.setParam('separating/zerohalf/maxcutcands', max(max_cuts, max_cuts_root))
        model.setParam('separating/zerohalf/maxrounds', max_rounds)
        model.setParam('separating/zerohalf/maxroundsroot', max_rounds_root)
        model.setParam('separating/zerohalf/maxsepacuts', max_cuts)
        model.setParam('separating/zerohalf/maxsepacutsroot', max_cuts_root)

        # Now the general cut and round parameters
        model.setParam("separating/maxroundsroot", max_rounds_root)
        model.setParam("separating/maxstallroundsroot", max_rounds_root)
        model.setParam("separating/maxcutsroot", max_cuts_root)

        model.setParam("separating/maxrounds", max_rounds)
        model.setParam("separating/maxstallrounds", 1)
        model.setParam("separating/maxcuts", max_cuts)

    def _init_scip_params(self, **init_scip_kwargs):
        seed = self.scip_seed % 2147483648  # SCIP seed range

        # set up randomization
        self.m.setBoolParam('randomization/permutevars', True)
        self.m.setIntParam('randomization/permutationseed', seed)
        self.m.setIntParam('randomization/randomseedshift', seed)

        # separators
        self._set_scip_separator_params(init_scip_kwargs['max_rounds_root'], 1, 10000, 1000, 10)

        # separation only at root node
        self.m.setIntParam('separating/maxrounds', 0)

        # no restart
        self.m.setIntParam('presolving/maxrestarts', 0)

        # if asked, disable presolving
        if not init_scip_kwargs['presolving']:
            self.m.setIntParam('presolving/maxrounds', 0)
            self.m.setIntParam('presolving/maxrestarts', 0)

        # if asked, disable separating (cuts)
        if not init_scip_kwargs['separating']:
            self.m.setIntParam('separating/maxroundsroot', 0)

        # if asked, disable conflict analysis (more cuts)
        if not init_scip_kwargs['conflict']:
            self.m.setBoolParam('conflict/enable', False)

        # if asked, disable primal heuristics
        if not init_scip_kwargs['heuristics']:
            self.m.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

    def set_seed(self, seed=None):
        if seed:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(self.seed)

    def reset(self, log_prefix):
        # create scip model
        self.m = scip.Model()
        if self.single_instance_file == 'all':
            instance_file = self.rng.choice(self.instances)
        else:
            instance_file = self.single_instance_file
        # instance_file = 'instance_9575.lp'
        logger.log(f"{log_prefix} instance_file: {instance_file}")
        instance_file = os.path.join(self.instance_file_path, instance_file)
        self.m.setIntParam('display/verblevel', 0)
        self.m.readProblem(instance_file)
        self.m.setRealParam('limits/time', self.scip_time_limit)
        self.m.setIntParam('timing/clocktype', 2)
        self._init_scip_params(**self.init_scip_kwargs)

        return instance_file

    def step(self, CutSel):
        # include cutsel
        self.m.includeCutsel(
            cutsel=CutSel,
            name="RL trained cutsel",
            desc="",
            priority=666666
        )

        # optimize the scip model
        self.m.optimize()

        # get statstics
        stats = {}
        stats['solving_time'] = self.m.getSolvingTime()
        stats['ntotal_nodes'] = self.m.getNTotalNodes()
        stats['primal_dual_gap'] = self.m.getGap()
        stats['primaldualintegral'] = self.m.getPrimalDualIntegral()

        # free problem
        self.m.freeProb()

        return stats

    def set_random_seed(self, seed):
        self.rng = np.random.RandomState(seed)
