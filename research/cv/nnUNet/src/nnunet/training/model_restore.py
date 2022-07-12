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

"""model restore module"""

import importlib
import pkgutil

import mindspore
from batchgenerators.utilities.file_and_folder_operations import load_pickle, join, isdir, subfolders

import src.nnunet
from src.nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer


def recursive_find_python_class(folder, trainer_name, current_module):
    """recursive find python class"""
    tr = None
    for _, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for _, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name,
                                                 current_module=next_current_module)
            if tr is not None:
                break

    return tr


def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer
    and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    search_in = join(src.nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="src.nnunet.training.network_training")

    if tr is None:
        # Fabian only. This will trigger searching for trainer classes in other repositories as well
        try:
            import meddec
            search_in = join(meddec.__path__[0], "model_training")
            tr = recursive_find_python_class([search_in], name, current_module="meddec.ckpt_training")
        except ImportError:
            pass

    if tr is None:
        raise RuntimeError(
            "Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
            "is not located there, please move it or change the code of restore_model. Your model "
            "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
            "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
                                          "Please make it so!"

    trainer = tr(*init)

    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_best_model_for_inference(folder):
    """load best model for inference"""
    checkpoint = join(folder, "model_best.ckpt")
    pkl_file = checkpoint + ".pkl"
    return restore_model(pkl_file, checkpoint, False)


def load_model_and_checkpoint_files(folder, folds=None, mixed_precision=None, checkpoint_name="model_best"):
    """
    used for if you need to ensemble the five models of a cross-validation.
    This will restore the model from the checkpoint in fold 0,
    load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them form disk each time).
    This is best used for inference and test prediction.
    """
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds.")
    trainer = restore_model(join(folds[0], "%s.ckpt.pkl" % checkpoint_name), fp16=mixed_precision)
    print("%s.ckpt.pkl" % checkpoint_name)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)
    all_best_model_files = [join(i, "%s.ckpt" % checkpoint_name) for i in folds]
    # mindspore need ckpt
    print("using the following model files: ", all_best_model_files)
    all_params = [mindspore.load_checkpoint(i) for i in all_best_model_files]
    return trainer, all_params
