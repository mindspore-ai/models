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

"""nnUNetTrainerV2"""

from collections import OrderedDict
from typing import Tuple

import mindspore

from mindspore import FixedLossScaleManager, nn, ops, Tensor
from mindspore._checkparam import Validator as validator
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_pickle, load_pickle, isfile

from sklearn.model_selection import KFold

from src.nnunet.network_architecture.generic_UNet import Generic_UNet
from src.nnunet.network_architecture.initialization import InitWeights_He
from src.nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from src.nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from src.nnunet.training.dataloading.dataset_loading import unpack_dataset
from src.nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from src.nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from src.nnunet.utilities.nd_softmax import softmax_helper
from src.nnunet.utilities.to_mindspore import maybe_to_mindspore

loss_scale = 1024.
loss_scale_manager = FixedLossScaleManager(loss_scale, False)
grad_scale = ops.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def gradient_scale(scale, grad):
    """gradient scale"""
    return grad * ops.cast(scale, ops.dtype(grad))


class CustomTrainOneStepCell(nn.TrainOneStepCell):
    """Custom Train One Step Cell"""

    def __init__(self, network, optimizer, sens=1.0):
        super(CustomTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.hyper_map = ops.HyperMap()
        self.reciprocal_sense = Tensor(1 / sens, mindspore.float32)

    def scale_grad(self, gradients):
        """scale grad"""
        gradients = self.hyper_map(ops.partial(grad_scale, self.reciprocal_sense), gradients)
        return gradients

    def construct(self, *inputs):
        """construct network"""
        loss = self.network(*inputs)
        sens = ops.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.scale_grad(grads)
        grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class WithLossCell(nn.Cell):
    """WithLossCell Class"""

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, y_0, y_1, y_2):
        """construct loss"""
        out = self._backbone(data)
        return self._loss_fn(out, y_0, y_1, y_2)


class WithEvalCell(nn.Cell):
    """WithEvalCell Class"""

    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = validator.check_value_type("add_cast_fp32", add_cast_fp32, [bool], self.cls_name)

    def construct(self, data, y0, y1, y2):
        """construct forward"""
        outputs = self._network(data)
        loss = self._loss_fn(outputs, y0, y1, y2)
        return loss, outputs


class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.was_initialized = False
        self.do_dummy_2D_aug = None
        self.max_num_epochs = 200
        self.initial_lr = 1e-3
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """initialize function"""
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            # wrap the loss for deep supervision
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True] + [bool(i < net_numpool - 1)  for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            self.net_with_criterion = WithLossCell(self.network, self.loss)

            self.train_net = CustomTrainOneStepCell(self.net_with_criterion, self.optimizer, loss_scale)
            self.train_net.set_train(True)
            self.eval_net = WithEvalCell(self.network, self.loss)
            self.eval_net.set_train(False)

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def g(self, x):
        """same result as lambda x:x"""
        return x

    def initialize_network(self):
        """
        initialize the network
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout
            norm_op = nn.BatchNorm3d
            self.one_hot = self.one_hot3D
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout
            norm_op = nn.BatchNorm2d
            self.one_hot = self.one_hot2D

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'alpha': 1e-2}

        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, self.g, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,)

        self.network.inference_apply_nonlin = softmax_helper

    def poly_lr(self, epoch, exponent=0.9):
        """plot the learning rate"""
        return self.initial_lr * (1 - epoch / self.max_num_epochs) ** exponent

    def initialize_optimizer_and_scheduler(self):
        """initialize optimizer and scheduler"""
        assert self.network is not None, "self.initialize_network must be called first"

        self.lr = []
        for i in range(0, self.max_num_epochs):
            self.lr.append(self.poly_lr(epoch=i))
        self.optimizer = nn.Adam(self.network.trainable_params(), self.initial_lr)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target_o):
        """run online evaluation"""
        output = output[0]
        return super().run_online_evaluation(output, target_o)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """validate"""
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True, file_name=None,
                                                         img_path: str = None,
                                                         covert_Ascend310_file: bool = True
                                                         ) -> Tuple[np.ndarray, np.ndarray]:
        """predict preprocess data return seg and softmax"""
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision,
                                                                       file_name=file_name,
                                                                       img_path=img_path,
                                                                       covert_Ascend310_file=True)
        self.network.do_ds = ds
        return ret

    def one_hot2D(self, labels, num_classes=3):
        """mindspore onehot 2D"""
        onehot = ops.OneHot()
        reshape = ops.Reshape()
        shape = ops.Shape()
        trans = ops.Transpose()
        labels = Tensor(labels).astype("int32")
        n, _, d, w = shape(labels)
        num_classes, on_value, off_value = num_classes, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        labels = trans(labels, (0, 2, 3, 1))
        output = onehot(labels, num_classes, on_value, off_value)
        output = reshape(output, (n, d, w, num_classes))
        output = trans(output, (0, 3, 1, 2))
        return output

    def one_hot3D(self, labels, num_classes=3):
        """mindspore onehot 3D"""
        onehot = ops.OneHot()
        reshape = ops.Reshape()
        shape = ops.Shape()
        trans = ops.Transpose()
        labels = Tensor(labels).astype("int32")
        n, _, d, w, h = shape(labels)
        num_classes, on_value, off_value = num_classes, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        labels = trans(labels, (0, 2, 3, 4, 1))
        output = onehot(labels, num_classes, on_value, off_value)
        output = reshape(output, (n, d, w, h, num_classes))
        output = trans(output, (0, 4, 1, 2, 3))

        return output

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=True):
        """run iteration"""
        data_dict = next(data_generator)
        data = data_dict['data']
        target_o = data_dict['target'][0]

        target = data_dict['target']

        data = maybe_to_mindspore(data)

        target_1 = Tensor(self.one_hot(target[1]), mindspore.float32)
        target_0 = Tensor(self.one_hot(target[0]), mindspore.float32)
        target_2 = Tensor(self.one_hot(target[2]), mindspore.float32)

        l = self.train_net(data, target_0, target_1, target_2)

        output = self.eval_net(data, target_0, target_1, target_2)

        if run_online_evaluation:
            self.run_online_evaluation(output[1], target_o)

        del target
        return l.asnumpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.

        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase rotation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        oringinally lr rate shuold be updated but we set it staticlly
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)
        """

        if epoch is None:
            ep = self.epoch
        else:
            ep = epoch

        self.print_to_log_file("lr:", np.round(self.lr[ep], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs

        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
