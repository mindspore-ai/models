# Copyright 2023 Huawei Technologies Co., Ltd
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
# =======================================================================================
""" serving server entrance module """
from mindspore_serving import server
from yolox.paraser import config


def start(cfg):
    serving_config = server.ServableStartConfig(servable_directory=cfg.servable_dir, servable_name=cfg.servable_name,
                                                device_ids=cfg.device_id, num_parallel_workers=1)
    server.start_servables(serving_config)
    server.start_grpc_server('%s:%s' % (cfg.ip, cfg.port))


if __name__ == '__main__':
    start(config)
