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

"""nnUNet 310 postprocess."""

import os
from copy import deepcopy
from typing import Union, Tuple

import argparse
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_pickle, isfile

from src.nnunet.configuration import RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD
from src.nnunet.preprocessing.preprocessing import resample_data_or_seg
from src.nnunet.evaluation.evaluator import evaluate_folder



def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """do separate z"""
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    "get lowres axis"
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def softmax(x, axis=None):
    """soft max result"""
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def get_file_ls(path, ls):
    """read file information"""
    for _, _, result_files in os.walk(path):
        for file in result_files:
            ls.append(file)
    return ls


def get_result_file_slice(filename):
    """get_result_file_slice"""
    left_id = 18
    right_id = 0
    for i in range(18, len(filename)):
        if filename[i] == "_":
            right_id = i
            break

    return filename[left_id:right_id]


def get_network_out_index(file_name):
    """get_network_out_index"""
    return file_name[-5]


def get_shape_slice(file_name):
    """get_shape_slice"""
    right_id = -4
    left_id = 0

    for i in range(right_id, -len(file_name), -1):
        if file_name[i] == '_':
            left_id = i
            break

    return file_name[left_id + 1:right_id]


def get_aggregated_nb_of_predictions_slice(file_name):
    """get_aggregated_nb_of_predictions_slice"""
    right_id = -4
    left_id = 0
    for i in range(right_id, -len(file_name), -1):
        if file_name[i] == 's':
            left_id = i
            break

    return file_name[left_id + 1:right_id]


def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing segmentations to nifto and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifto export
    There is a problem with python process communication that prevents us from communicating obejcts
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    print("segmentation_export")
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        segmentation_softmax = np.load(segmentation_softmax)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')


    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)

    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
                     bbox[1][0]:bbox[1][1],
                     bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)


def post_process2d():
    # prepare data information to recovery
    if not os.path.exists('scripts/result_Files/inferTs'):
        os.mkdir('scripts/result_Files/inferTs')
    result_files_ls = []
    data = []
    location_ls = []
    shape_ls = []
    slicer_ls = []
    aggregated_nb_of_predictions_ls = []
    dct_ls = []

    result_root = "scripts/result_Files"
    location_root = "preprocess_Result/location"
    slicer_root = "preprocess_Result/slicer"
    shape_root = "preprocess_Result/data_shape"
    aggregated_root = "preprocess_Result/aggregated_nb_of_predictions"
    dct_root = "preprocess_Result/dct"

    result_files_ls = get_file_ls(result_root, result_files_ls)
    location_ls = get_file_ls(location_root, location_ls)
    shape_ls = get_file_ls(shape_root, shape_ls)
    slicer_ls = get_file_ls(slicer_root, slicer_ls)
    aggregated_nb_of_predictions_ls = get_file_ls(aggregated_root, aggregated_nb_of_predictions_ls)
    dct_ls = get_file_ls(dct_root, dct_ls)
    print(aggregated_nb_of_predictions_ls)
    aggregated_nb_of_predictions_ls.sort(key=lambda x: int(x.split('.')[0].split("predictions")[1]))

    for case in dct_ls:
        softmax_pred = []
        result_file = []
        data_shape = []
        lb_x = 0
        ub_x = 0
        lb_y = 0
        ub_y = 0
        slicer = []
        aggregated_results = []
        # traverse the document
        for f in aggregated_nb_of_predictions_ls:
            if case[:15] == f[:15]:
                aggregated_nb_of_predictions = np.fromfile(os.path.join(aggregated_root, f), dtype=np.float32)
                for result_file in result_files_ls:
                    # get id
                    if result_file[:16] == f[:16] and get_result_file_slice(
                            result_file) == get_aggregated_nb_of_predictions_slice(f) and get_network_out_index(
                                result_file) == '0':
                        data.append(result_file)
                print("---------------")
                print("bboxes,", data)
                print("aggregated", f)

                # softmax
                # Inference results include deep supervision feature map
                # return first bbox returned
                data_shape_Flag = False
                aggregated_results, data_shape, result_file = get_ori_shape(aggregated_results, data, data_shape,
                                                                            data_shape_Flag, lb_x, lb_y, location_ls,
                                                                            location_root, result_root, shape_ls,
                                                                            shape_root, ub_x, ub_y)

                # get slicer
                for slicer_file in slicer_ls:
                    if result_file[:16] == slicer_file[:16] and get_result_file_slice(result_file) == get_shape_slice(
                            slicer_file):
                        slicer = np.load(os.path.join(slicer_root, slicer_file), allow_pickle=True)
                        break

                # get nb_of_predictions
                aggregated_nb_of_predictions.shape = 3, data_shape[1], data_shape[2]
                class_probabilities = aggregated_results[tuple(slicer[0])] / aggregated_nb_of_predictions[
                    tuple(slicer[0])]
                softmax_pred.append(class_probabilities[None])
                data.clear()

            # find dict
        properties_dict_for_export = np.load(os.path.join(dct_root, case), allow_pickle=True).item()
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        save_segmentation_nifti_from_softmax(softmax_pred,
                                             'scripts/result_Files/inferTs/' + result_file[:15] + '.nii.gz',
                                             properties_dict_for_export)

    # Fetch folder for verification


def get_ori_shape(aggregated_results, data, data_shape, data_shape_Flag, lb_x, lb_y, location_ls, location_root,
                  result_root, shape_ls, shape_root, ub_x, ub_y):
    result_file = []
    for result_file in data:
        # get original shape

        for shape_file in shape_ls:
            if result_file[:16] == shape_file[:16] and get_result_file_slice(
                    result_file) == get_shape_slice(shape_file) and not data_shape_Flag:
                # get correct file
                print("shape_file", shape_file)
                data_shape = np.fromfile(os.path.join(shape_root, shape_file), dtype=np.int)
                aggregated_results = np.zeros(shape=(3, data_shape[1], data_shape[2]), dtype=np.float32)
                print("result_file", get_result_file_slice(result_file))
                print("shape_file", get_shape_slice(shape_file))
                data_shape_Flag = True
                break

        result_data = np.fromfile(os.path.join(result_root, result_file), dtype=np.float32)
        result_data.shape = 366, 3, 56, 40
        # softmax
        result_data = softmax(result_data, axis=1)[0]
        result_data = np.expand_dims(result_data, 0)

        # get location
        for location_file in location_ls:
            if result_file[:18] == location_file[:18] and get_result_file_slice(
                    result_file) == get_shape_slice(location_file):
                location = np.fromfile(os.path.join(location_root, location_file), dtype=np.int)
                lb_x, ub_x, lb_y, ub_y = location
                break
        aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += result_data[0]
    return aggregated_results, data_shape, result_file


def post_process3d():
    """prepare data information to recovery"""
    if not os.path.exists('scripts/result_Files/inferTs'):
        os.mkdir('scripts/result_Files/inferTs')
    result_files_ls = []
    data = []
    location_ls = []
    shape_ls = []
    slicer_ls = []
    aggregated_nb_of_predictions_ls = []
    dct_ls = []

    result_root = "scripts/result_Files"
    location_root = "preprocess_Result/location"
    slicer_root = "preprocess_Result/slicer"
    shape_root = "preprocess_Result/data_shape"
    aggregated_root = "preprocess_Result/aggregated_nb_of_predictions"
    dct_root = "preprocess_Result/dct"

    result_files_ls = get_file_ls(result_root, result_files_ls)
    location_ls = get_file_ls(location_root, location_ls)
    shape_ls = get_file_ls(shape_root, shape_ls)
    slicer_ls = get_file_ls(slicer_root, slicer_ls)
    aggregated_nb_of_predictions_ls = get_file_ls(aggregated_root, aggregated_nb_of_predictions_ls)
    dct_ls = get_file_ls(dct_root, dct_ls)

    for root, _, files in os.walk("preprocess_Result/aggregated_nb_of_predictions"):
        result_file = []
        aggregated_results = []
        lb_x = 0
        ub_x = 0
        lb_y = 0
        ub_y = 0
        lb_z = 0
        ub_z = 0
        data_shape = 0
        slicer = []
        for f in files:

            aggregated_nb_of_predictions = np.fromfile(os.path.join(root, f))
            for result_file in result_files_ls:

                if result_file[:16] == f[:16] and result_file[19] == '0':
                    data.append(result_file)

            # softmax
            # Inference results include deep supervision feature map
            # return first bbox returned
            data_shape_Flag = False
            for result_file in data:
                # get original shape

                for shape_file in shape_ls:
                    if result_file[:16] == shape_file[:16] and not data_shape_Flag:
                        data_shape = np.fromfile(os.path.join(shape_root, shape_file), dtype=np.int)
                        aggregated_results = np.zeros(shape=(3, data_shape[1], data_shape[2], data_shape[3]),
                                                      dtype=np.float32)
                        data_shape_Flag = True
                        break

                result_data = np.fromfile(os.path.join(result_root, result_file), dtype=np.float32)
                result_data.shape = 9, 3, 40, 56, 40
                # softmax
                result_data = softmax(result_data, axis=1)[0]
                result_data = np.expand_dims(result_data, 0)

                # get location
                for location_file in location_ls:
                    if result_file[:18] == location_file[:18]:
                        location = np.fromfile(os.path.join(location_root, location_file), dtype=np.int)
                        lb_x, ub_x, lb_y, ub_y, lb_z, ub_z = location

                # padding

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += result_data[0]

            # get slicer
            for slicer_file in slicer_ls:
                if result_file[:16] == slicer_file[:16]:
                    slicer = np.load(os.path.join(slicer_root, slicer_file), allow_pickle=True)

            # get nb_of_predictions
            for aggregated_nb_of_predictions_file in aggregated_nb_of_predictions_ls:
                if result_file[:16] == aggregated_nb_of_predictions_file[:16]:
                    aggregated_nb_of_predictions = np.fromfile(
                        os.path.join(aggregated_root, aggregated_nb_of_predictions_file), dtype=np.float32)
                    aggregated_nb_of_predictions.shape = 3, data_shape[1], data_shape[2], data_shape[3]

            class_probabilities = aggregated_results[tuple(slicer[0])] / aggregated_nb_of_predictions[tuple(slicer[0])]
            # find dict
            result_dict = []
            for dct_file in dct_ls:

                if result_file[:15] == dct_file[:15]:
                    result_dict = np.load(os.path.join(dct_root, dct_file), allow_pickle=True).item()

                    break

            result_properties_dict = result_dict
            print(dict)

            save_segmentation_nifti_from_softmax(class_probabilities,
                                                 'scripts/result_Files/inferTs/' + result_file[:15] + '.nii.gz',
                                                 result_properties_dict)
            data.clear()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("network", default="nnUNet_3d_fullres")
    args = parser.parse_args()

    if args.network == "nnUNet_2d":
        post_process2d()
    else:
        post_process3d()

    evaluate_folder('src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/labelsVal',
                    'scripts/result_Files/inferTs', (0, 1, 2))
