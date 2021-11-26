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
import os

from mindspore import save_checkpoint


def save_ckpt(net, domain_specific_nets, save_path, args, iteration, epoch, pattern):
    if pattern == 1:
        save_checkpoint(net, os.path.join(save_path, args.save_folder, args.save_file_RL) +
                        '_epoch' + repr(epoch) + '_iter' + repr(iteration) + '.ckpt')

        for curr_domain, domain_specific_net in enumerate(domain_specific_nets):
            save_checkpoint(domain_specific_net,
                            os.path.join(save_path, args.save_folder, args.save_domain_dir,
                                         'RL_epoch' + repr(epoch) + '_iter' + repr(
                                             iteration) + '_' + str(curr_domain) + '.ckpt'))
    elif pattern == 2:
        save_checkpoint(net, os.path.join(save_path,
                                          args.save_folder, args.save_file_RL) + 'epoch' + repr(epoch) + '.ckpt')

        for curr_domain, domain_specific_net in enumerate(domain_specific_nets):
            save_checkpoint(domain_specific_net,
                            os.path.join(save_path, args.save_folder, args.save_domain_dir,
                                         'RL_epoch' + repr(epoch) + '_' + str(curr_domain) + '.ckpt'))
    else:
        save_checkpoint(net, os.path.join(os.path.join(save_path, args.save_folder, args.save_file_RL) + '.ckpt'))
        if args.multidomain:
            for curr_domain, domain_specific_net in enumerate(domain_specific_nets):
                save_checkpoint(domain_specific_net,
                                os.path.join(save_path, args.save_folder, args.save_domain_dir,
                                             '_' + str(curr_domain) + '.ckpt'))
