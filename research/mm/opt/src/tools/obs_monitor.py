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
"""Obs Monitor."""
import os
import re
import time
import moxing as mox
# from moxing.framework.file import file_io

from .logger import LOGGER


# mox.file.set_auth(is_secure=False)
# file_io.NUMBER_OF_PROCESSES = 10

def _sort_obs_ckpt(obs_path):
    """Sorts checkpoint files by name."""
    file_list = mox.file.list_directory(obs_path)
    ckpt_list = [x for x in file_list if x.endswith(".ckpt")]
    if not ckpt_list:
        return None

    # sort the ckpt_file_list according to the ckpt name.
    fake_ckpt_list = []
    for ckpt in ckpt_list:
        if ckpt.count("_") == 2:
            fake_ckpt_list.append(ckpt)
        else:
            prefix, suffix = ckpt.split("-")
            new_ckpt = prefix + "_0" + "-" + suffix
            fake_ckpt_list.append(new_ckpt)

    fake_ckpt_list.sort(key=lambda x: (-int(re.split(r"[_|\-|.]", x)[1]),
                                       -int(re.split(r"[_|\-|.]", x)[2]), -int(re.split(r"[_|\-|.]", x)[3])))
    sorted_ckpt_list = [x.replace("_0", "") for x in fake_ckpt_list]
    return sorted_ckpt_list


class ObsUploader:
    """Obs Uploader"""
    def __init__(self, bucket_dir, max_ckpt=5, retry=3, retry_time=10, interval_num=256, interval_time=90):
        self.bucket_dir = bucket_dir  # s3://muti-modal/ckpt/
        self.max_ckpt = max_ckpt
        self.retry = retry
        self.retry_time = retry_time
        self.interval_num = interval_num
        self.interval_time = interval_time

    def upload_ckpt(self, local_file_path):
        """Upload Ckpt"""
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        success = False

        obs_dir = os.path.join(self.bucket_dir, "rank_" + str(rank_id))  # s3://muti-modal/ckpt/rank_0
        file_name = local_file_path.split('/')[-1]  # /cache/ckpt/rank_0/OPT-4590_1.ckpt

        obs_file_path = os.path.join(obs_dir, file_name)
        # s3://muti-modal/ckpt/upload_log
        except_log_dir = os.path.join(self.bucket_dir, "upload_log")
        if not mox.file.exists(except_log_dir):
            mox.file.mk_dir(except_log_dir)

        except_file_path = os.path.join(
            except_log_dir, f"except_upload_rank_{rank_id}_{file_name}_{int(time.time())}.log")

        except_info = ""

        # sleep due to restriction of obs
        sleep_time = int(rank_id) // self.interval_num * self.interval_time
        if sleep_time > 0:
            LOGGER.info(
                "rank_%d waits %d s before uploading.", rank_id, sleep_time)
            time.sleep(sleep_time)

        LOGGER.info(
            "rank_%d : start uploading %s to %s.", rank_id, local_file_path, obs_file_path)
        for i in range(self.retry + 1):
            try:
                if not mox.file.exists(obs_dir):
                    mox.file.mk_dir(obs_dir)

                file_list = mox.file.list_directory(obs_dir)
                ckpt_file = [x for x in file_list if x.endswith(".ckpt")]
                if len(ckpt_file) >= self.max_ckpt:
                    oldest_ckpt = _sort_obs_ckpt(obs_dir)[-1]
                    mox.file.remove(os.path.join(obs_dir, oldest_ckpt))

                start = time.time()
                mox.file.copy(local_file_path, obs_file_path)
                end = time.time()
                if mox.file.exists(obs_file_path):
                    success = True
                    LOGGER.info(
                        "rank_%d: uploading %s to %s cost %d s.",
                        rank_id, local_file_path, obs_file_path, end-start)
                    break
            except RuntimeError as e:
                if i < self.retry:
                    loginfo = e.__str__()
                    loginfo += " rank_%d: uploading %s to %s failed: retry %d /%d." %\
                              (rank_id, local_file_path, obs_file_path, i+1, self.retry)
                    LOGGER.info(loginfo)

                    time.sleep(self.retry_time)
                    self.retry_time = self.retry_time + 10
                else:
                    except_info = e.__str__()

        if not success:
            mox.file.append(except_file_path, f"{except_info}. rank_{rank_id}: uploading {local_file_path} to "
                                              f"{obs_file_path} failed.\n")


class ObsRestorer:
    """Obs Restorer"""
    def __init__(self, bucket_dir, retry=3, retry_time=30, interval_num=256, interval_time=90):
        self.bucket_dir = bucket_dir
        self.retry = retry
        self.retry_time = retry_time
        self.interval_num = interval_num
        self.interval_time = interval_time

    def retry_rank0(self, log_dir, latest_corresponding_ckpt, return_flag):
        """retry_rank0"""

        for i in range(self.retry + 1):
            try:
                # list rank dir
                obs_dirs = mox.file.list_directory(self.bucket_dir)
                rank_dirs = [x for x in obs_dirs if x.startswith(
                    "rank_")]  # [rank_0, rank_1]
                rank_first_path = os.path.join(
                    self.bucket_dir, rank_dirs[0])
                sorted_ckpt_list = _sort_obs_ckpt(rank_first_path)

                if sorted_ckpt_list is None:
                    mox.file.append(os.path.join(log_dir, f"except_find_latest_ckpt.log"),
                                    f"obs restoring: find no checkpoint file in {rank_first_path}.")
                    return None

                for ckpt in sorted_ckpt_list:
                    flag = True
                    for dir_ in rank_dirs:
                        obs_path = os.path.join(self.bucket_dir, dir_, ckpt)
                        if not mox.file.exists(obs_path):
                            flag = False
                            break
                    if flag:
                        latest_corresponding_ckpt = ckpt
                        break

                if latest_corresponding_ckpt is None:
                    mox.file.append(os.path.join(log_dir, f"except_find_latest_ckpt.log"),
                                    f"obs restoring: find no corresponding checkpoint file.")
                    return_flag = True
                    break
                else:
                    mox.file.append(os.path.join(log_dir, "restore_" + latest_corresponding_ckpt),
                                    f"obs restoring: latest corresponding ckpt is {latest_corresponding_ckpt}.")
                    break
            except RuntimeError:
                if i < self.retry:
                    time.sleep(self.retry_time)
                else:
                    mox.file.append(os.path.join(log_dir, f"except_find_latest_ckpt.log"),
                                    f"obs restoring: find no corresponding checkpoint file.")
                    return_flag = True
                    break

        return return_flag, latest_corresponding_ckpt


    def restore_ckpt(self, local_ckpt_dir):
        """Downloads the latest corresponding checkpoint file to local device from obs."""
        # get rank id

        log_dir = os.path.join(self.bucket_dir, "restore_log")  # s3://muti-modal/ckpt/restore_log

        # rank_id = os.getenv('RANK_ID')
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        latest_corresponding_ckpt = None

        if not mox.file.exists(log_dir):
            mox.file.mk_dir(log_dir)
        return_flag = False
        # Looking for the latest corresponding ckpt, rank 0 try three times.

        if int(rank_id) == 0:
            return_flag, latest_corresponding_ckpt = self.retry_rank0(log_dir, latest_corresponding_ckpt, return_flag)
        else:
            time.sleep(int(rank_id) % 8 + 1)
            log_files = mox.file.list_directory(log_dir)
            while not log_files:
                time.sleep(int(rank_id) % 4 + 1)
                log_files = mox.file.list_directory(log_dir)

            # just one file
            for log_file in log_files:
                if log_file.startswith("except_"):
                    return_flag = True
                    break
                elif log_file.startswith("restore"):
                    latest_corresponding_ckpt = log_file[8:]
                    break

        if return_flag:
            return None
        # sleep due to restriction of obs
        sleep_time = int(rank_id) // self.interval_num * self.interval_time
        if sleep_time > 0:
            LOGGER.info(
                "rank_%d waits %ds before restoring.", rank_id, sleep_time)
            time.sleep(sleep_time)

        # find obs ckpt
        obs_file_path = os.path.join(
            self.bucket_dir, "rank_" + str(rank_id), latest_corresponding_ckpt)
        if not mox.file.exists(obs_file_path):
            mox.file.append(os.path.join(self.bucket_dir, f"except_restore_rank_{rank_id}.log"),
                            f"rank_{rank_id} obs restoring: {obs_file_path} does not exist.")
            return None

        # download ckpt from obs to local dir.
        local_file_path = os.path.join(
            local_ckpt_dir, latest_corresponding_ckpt)
        success = False
        except_info = ""
        LOGGER.info(
            "rank_%d: start restoring %s to %s.",
            rank_id, obs_file_path, local_file_path)
        for i in range(self.retry + 1):
            try:
                start = time.time()
                mox.file.copy(obs_file_path, local_file_path)
                end = time.time()
                if os.path.exists(local_file_path):
                    success = True
                    LOGGER.info(
                        "rank_%d: restoring %s to %s cost %d s.",
                        rank_id, obs_file_path, local_file_path, end-start)
                    break

            except RuntimeError as e:
                if i < self.retry:
                    loginfo = e.__str__()
                    loginfo += " rank_%d: restoring %s to %s failed: retry %d/%d ." % \
                                (rank_id, obs_file_path, local_file_path, i + 1, self.retry)
                    LOGGER.info(loginfo)
                    time.sleep(self.retry_time)
                else:
                    except_info = e.__str__()

        if not success:
            mox.file.append(os.path.join(log_dir, f"except_restore_rank_{rank_id}.log"),
                            f"{except_info}. rank_{rank_id} obs restoring: restoring from obs failed.")
            local_file_path = None
        else:
            success_file_path = os.path.join(
                log_dir, f"success_restore_rank_{rank_id}.log")
            mox.file.append(success_file_path,
                            f"rank_{rank_id} obs restoring: restoring {latest_corresponding_ckpt} from obs succeed.")

        return local_file_path


class SOMARestorer:
    """SOMA Restorer"""
    def __init__(self, obs_dir="s3://mindspore-file/soma/gpt_2048", retry=3,
                 retry_time=10, interval_num=1024, interval_time=30):
        self.obs_dir = obs_dir
        self.retry = retry
        self.retry_time = retry_time
        self.interval_num = interval_num
        self.interval_time = interval_time

    def restore_soma(self, soma_local_dir):
        """restore soma"""
        # get rank id
        log_dir = os.path.join(self.obs_dir, "restore_log")
        rank_id = os.getenv('RANK_ID')
        if not mox.file.exists(log_dir):
            mox.file.mk_dir(log_dir)
        soma_obs_dir = os.path.join(self.obs_dir, f"rank_{rank_id}")

        # sleep due to restriction of obs
        sleep_time = int(rank_id) // self.interval_num * self.interval_time
        if sleep_time > 0:
            LOGGER.info(
                "rank_%d waits %d s before restoring soma.",
                rank_id, sleep_time)
            time.sleep(sleep_time)

        # download soma from obs.
        success = False
        except_info = ""
        LOGGER.info(
            "rank_%d: start restoring %s to %s.",
            rank_id, soma_obs_dir, soma_local_dir)
        for i in range(self.retry + 1):
            try:
                start = time.time()
                mox.file.copy_parallel(soma_obs_dir, soma_local_dir)
                end = time.time()
                success = True
                LOGGER.info(
                    "rank_%d: restoring %s to %s cost %d s.",
                    rank_id, soma_obs_dir, soma_local_dir, end - start)
                break

            except RuntimeError as e:
                if i < self.retry:
                    loginfo = e.__str__() + \
                              " rank_{}: restoring {} to {} failed: retry {} / {}.".\
                                  format(rank_id, soma_obs_dir, soma_local_dir, i + 1, self.retry)
                    LOGGER.info(loginfo)
                    time.sleep(self.retry_time)
                else:
                    except_info = e.__str__()

        if not success:
            mox.file.append(os.path.join(log_dir, f"except_restore_rank_{rank_id}_soma.log"),
                            f"{except_info}. rank_{rank_id} obs restoring: restoring soma from obs failed.")
            soma_local_dir = None
        else:
            success_file_path = os.path.join(
                log_dir, f"success_restore_rank_{rank_id}_soma.log")
            mox.file.append(success_file_path,
                            f"rank_{rank_id} obs restoring: restoring {soma_obs_dir} from obs succeed.")

        return soma_local_dir
