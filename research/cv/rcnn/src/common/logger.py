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
"""
logger utils
"""
import logging
import os.path
import traceback
import colorlog

from src import paths
from mindspore.communication import get_rank

def wrap_trace_info(message):
    trace_info = traceback.extract_stack()[-3]
    filename = trace_info.filename.split('/')[-1]
    line_number = trace_info.lineno
    name = trace_info.name
    if name == '<module>':
        message = "%s(%s): %s" % (filename, line_number, message)
    else:
        message = "%s(%s)->%s(): %s" % (filename, line_number, name, message)
    return message


class Logger:
    """
    logger utils
    """

    def __init__(self, name='main', debug=True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        device_num = int(os.getenv('RANK_SIZE', '1'))
        if device_num > 1:
            rank_id = get_rank()
        elif device_num == 1:
            rank_id = 0
        log_dir = paths.Log.root + str(rank_id)

        # set handlers
        if not self.logger.handlers:

            # console handler
            console_handler = logging.StreamHandler()
            if debug:
                console_handler.setLevel(logging.DEBUG)
            else:
                console_handler.setLevel(logging.INFO)
            # color & format
            log_colors_config = {
                'DEBUG': 'white',
                'INFO': 'blue',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
            formatter = colorlog.ColoredFormatter(
                fmt='%(log_color)s[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors=log_colors_config
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # file handlers
            formatter = logging.Formatter(
                fmt='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            # check dirs
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            filename = name + '.txt'
            file_handler = logging.FileHandler(os.path.join(log_dir, filename), encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

            # debug file handlers
            debug_filename = name + '_DEBUG.txt'
            debug_file_handler = logging.FileHandler(os.path.join(log_dir, debug_filename), encoding='utf-8')
            debug_file_handler.setFormatter(formatter)
            debug_file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(debug_file_handler)

    def get_logger(self):
        return self.logger

    def debug(self, message):
        self.logger.debug(wrap_trace_info(message))

    def info(self, message):
        self.logger.info(wrap_trace_info(message))

    def warning(self, message):
        self.logger.warning(wrap_trace_info(message))

    def error(self, message, exc_info=False):
        self.logger.error(wrap_trace_info(message), exc_info=exc_info)

    def critical(self, message, exit_direct=True, exc_info=False):
        self.logger.critical(wrap_trace_info(message), exc_info=exc_info)
        if exit_direct:
            exit(1)
