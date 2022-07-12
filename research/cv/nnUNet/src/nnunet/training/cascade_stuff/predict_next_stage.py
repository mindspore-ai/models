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

"""predict next stage"""

from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, os, join, pardir
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from src.nnunet.preprocessing.preprocessing import resample_data_or_seg


def resample_and_save(predicted, target_shape, output_file, force_separate_z=False,
                      interpolation_order=1, interpolation_order_z=0):
    """resample and save function"""
    if isinstance(predicted, str):
        assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
                                  "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(predicted)
        predicted = np.load(predicted)
        os.remove(del_file)

    predicted_new_shape = resample_data_or_seg(predicted, target_shape, False, order=interpolation_order,
                                               do_separate_z=force_separate_z, order_z=interpolation_order_z)
    seg_new_shape = predicted_new_shape.argmax(0)
    np.savez_compressed(output_file, data=seg_new_shape.astype(np.uint8))


def predict_next_stage(trainer, stage_to_be_predicted_folder):
    """predict next stage function"""
    output_folder = join(pardir(trainer.output_folder), "pred_next_stage")
    maybe_mkdir_p(output_folder)

    if 'segmentation_export_params' in trainer.plans.keys():
        force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
        interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0

    export_pool = Pool(2)
    results = []

    for pat in trainer.dataset_val.keys():
        print(pat)
        data_file = trainer.dataset_val[pat]['data_file']
        data_preprocessed = np.load(data_file)['data'][:-1]

        predicted_probabilities = trainer.predict_preprocessed_data_return_seg_and_softmax(
            data_preprocessed, do_mirroring=trainer.data_aug_params["do_mirror"],
            mirror_axes=trainer.data_aug_params['mirror_axes'], mixed_precision=trainer.fp16)[1]

        data_file_nofolder = data_file.split("/")[-1]
        data_file_nextstage = join(stage_to_be_predicted_folder, data_file_nofolder)
        data_nextstage = np.load(data_file_nextstage)['data']
        target_shp = data_nextstage.shape[1:]
        output_file = join(output_folder, data_file_nextstage.split("/")[-1][:-4] + "_segFromPrevStage.npz")

        if np.prod(predicted_probabilities.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
            np.save(output_file[:-4] + ".npy", predicted_probabilities)
            predicted_probabilities = output_file[:-4] + ".npy"

        results.append(export_pool.starmap_async(resample_and_save, [(predicted_probabilities, target_shp, output_file,
                                                                      force_separate_z, interpolation_order,
                                                                      interpolation_order_z)]))

    _ = [i.get() for i in results]
    export_pool.close()
    export_pool.join()
