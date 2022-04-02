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
"""eval callbacks"""
import os
import stat
import shutil
from got10k.experiments import ExperimentOTB
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, \
                                        load_param_into_net,\
                                        save_checkpoint


class EvalCallBack(Callback):
    """
    Evaluation callback when training

    """
    def __init__(self, net, dataset, start_epoch=0,
                 end_epoch=50, save_path=None, interval=1):
        self.network = net
        self.start_epoch = start_epoch
        self.save_path = save_path
        self.interval = interval
        self.experiment = ExperimentOTB(dataset, version=2013)
        self.best_prec_score = 0
        self.best_succ_score = 0
        self.best_succ_rate = 0
    def epoch_end(self, run_context):
        """Callback when epoch end."""
        if os.path.exists('./results'):
            self.remove_ckpoint_file('./results')
            if os.path.exists('./reports'):
                self.remove_ckpoint_file('./reports')
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        ck_path = './models/siamfc/'+'SiamFC-'+str(cur_epoch)+'_6650.ckpt'
        print(ck_path, flush=True)
        if cur_epoch >= self.start_epoch:
            if (cur_epoch - self.start_epoch) % self.interval == 0:
                load_param_into_net(self.network.network, load_checkpoint(ck_path), strict_load=True)
                prec_score, succ_score, succ_rate = self.inference()
                if (prec_score >= self.best_prec_score) and (succ_score >= self.best_succ_score):
                    self.best_prec_score = prec_score
                    self.best_succ_score = succ_score
                    self.best_succ_rate = succ_rate
                    save_checkpoint(self.network.network, "best.ckpt")
                print("Best result:  prec_score: {}, succ_score: {}. "
                      "succ_rate: {}.".format(self.best_prec_score,
                                              self.best_succ_score,
                                              self.best_succ_rate), flush=True)
    def inference(self):
        """inference function"""
        self.experiment.run(self.network, visualize=False)
        results = self.experiment.report([self.network.name])
        prec_score = results[self.network.name]['overall']['precision_score']
        succ_score = results[self.network.name]['overall']['success_score']
        succ_rate = results[self.network.name]['overall']['success_rate']
        return float(prec_score), float(succ_score), float(succ_rate)

    def remove_ckpoint_file(self, file_name):
        """Remove the specified file."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            shutil.rmtree(file_name)
        except OSError:
            print("OSError, failed to remove the older ckpt file %s.", file_name, flush=True)
        except ValueError:
            print("ValueError, failed to remove the older ckpt file %s.", file_name, flush=True)
