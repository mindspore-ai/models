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

"""eval module"""

import argparse
import os

from batchgenerators.utilities.file_and_folder_operations import join, isdir

from mindspore import context
from mindspore.communication import init
from mindspore.context import ParallelMode

from src.nnunet.inference.predict import predict_from_folder
from src.nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, \
    default_trainer
from src.nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

def do_eval(parser):
    """eval logic according to parser logic"""
    device_id = int(os.getenv('DEVICE_ID'))
    run_distribute = int(os.getenv('distribute'))
    if run_distribute == 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        context.set_context(device_id=device_id)  # set device_id
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        context.set_context(device_id=device_id)  # set device_id
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    step_size = args.step_size
    overwrite_existing = args.overwrite_existing
    mode = args.mode
    all_in_gpu = args.all_in_gpu
    model = args.model
    trainer_class_name = args.trainer_class_name
    cascade_trainer_class_name = args.cascade_trainer_class_name
    task_name = args.task_name
    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)
    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], \
        "-m must be 2d, 3d_lowres, 3d_fullres or " \
        "3d_cascade_fullres"
    if lowres_segmentations == "None":
        lowres_segmentations = None
    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")
    # we need to catch the case where model is 3d cascade fullres and the low resolution folder has not been set.
    # In that case we need to try and predict with 3d low res first
    if model == "3d_cascade_fullres" and lowres_segmentations is None:
        print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
        assert part_id == 0 and num_parts == 1, "please run the 3d_lowres inference first"
        model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" +
                                 args.plans_identifier)
        assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
        lowres_output_folder = join(output_folder, "3d_lowres_predictions")
        predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False,
                            num_threads_preprocessing, num_threads_nifti_save, None, part_id, num_parts,
                            not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not args.disable_mixed_precision,
                            step_size=step_size,
                            img_path=args.img_path,
                            covert_Ascend310_file=args.covert_Ascend310_file
                            )
        lowres_segmentations = lowres_output_folder
        print("3d_lowres done")
    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name
    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                             args.plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
    print("args.chk", args.chk)
    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not args.disable_mixed_precision,
                        step_size=step_size, checkpoint_name=args.chk,
                        img_path=args.img_path,
                        covert_Ascend310_file=args.covert_Ascend310_file)

def main():
    """eval logic"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder',
                        help="Must contain all modalities for each patient in the correct", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default=default_plans_identifier, required=True)
    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the nnUNetTrainer used for 2D U-Net, full resolution 3D U-Net and low resolution',
                        required=False,
                        default=default_trainer)
    parser.add_argument('-ctr', '--cascade_trainer_class_name',
                        help="Trainer class name used for predicting the 3D full resolution U-Net part of the cascade."
                             "Default is %s" % default_cascade_trainer, required=False,
                        default=default_cascade_trainer)
    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres",
                        default="3d_fullres", required=False)
    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)
    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="folds to use for prediction. ")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax")
    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None')
    parser.add_argument("--part_id", type=int, required=False, default=0)
    parser.add_argument("--num_parts", type=int, required=False, default=1,
                        help="Used to parallelize the prediction of "
                             "the folder over several GPUs.")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int)
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int,)
    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. "
                             "Speeds up inference by roughly factor 4 (2D) or 8 (3D)")
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")
    parser.add_argument("--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument("--all_in_gpu", type=bool, default=None, required=False)
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    parser.add_argument('-chk',
                        help='checkpoint name, default: model_best',
                        required=False,
                        default='model_best')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False)
    parser.add_argument("--img_path", type=str, required=False,
                        default="./src/nnunet/preprocess_Result",
                        help="310 bin file_out_put")
    parser.add_argument("--covert_Ascend310_file", type=bool, required=False,
                        default=True,
                        help="whether covert 310_file")

    do_eval(parser)




if __name__ == "__main__":
    main()
