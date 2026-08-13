"""Training, evaluation, checkpoint, and result helpers."""

# A training entry point naturally owns the loss, optimiser, gradient
# function, early-stopping state, and per-epoch statistics.
# pylint: disable=too-many-locals

import json
import os
import random
import time

import mindspore as ms
from mindspore import nn, ops
import numpy as np

from .metrics import calculate_metrics


def configure_runtime(config):
    """Configure deterministic PYNATIVE execution for dynamic prompts."""
    context = {
        "mode": ms.PYNATIVE_MODE,
        "device_target": config.device_target,
    }
    if config.device_target != "CPU":
        context["device_id"] = config.device_id
    ms.set_context(**context)
    ms.set_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)


def attach_dataset_metadata(config, source):
    """Add data-dependent model dimensions to the configuration."""
    config.variable_names = source.variable_names
    config.enc_in = len(source.variable_names)
    config.target_index = source.target_index
    return config


def select_target(config, prediction, target):
    """Select all channels for M/S and only the target channel for MS."""
    feature_start = -1 if config.features == "MS" else 0
    prediction = prediction[:, -config.pred_len :, feature_start:]
    target = target[:, -config.pred_len :, feature_start:]
    return prediction, target


def evaluate_loss(config, model, dataset):
    """Return mean validation MSE and MAE."""
    loss_fn = nn.MSELoss()
    model.set_train(False)
    losses = []
    absolute_errors = []
    for batch_x, batch_y, _, _ in dataset.create_tuple_iterator(
        num_epochs=1
    ):
        prediction, target = select_target(
            config, model(batch_x), batch_y
        )
        losses.append(float(loss_fn(prediction, target).asnumpy()))
        absolute_errors.append(
            float(ops.mean(ops.abs(prediction - target)).asnumpy())
        )
    return float(np.mean(losses)), float(np.mean(absolute_errors))


def predict(config, model, dataset):
    """Collect predictions, targets, and historical input windows."""
    model.set_train(False)
    predictions = []
    targets = []
    histories = []
    for batch_x, batch_y, _, _ in dataset.create_tuple_iterator(
        num_epochs=1
    ):
        prediction, target = select_target(
            config, model(batch_x), batch_y
        )
        predictions.append(prediction.asnumpy())
        targets.append(target.asnumpy())
        histories.append(batch_x.asnumpy())
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(histories, axis=0),
    )


def load_trainable_checkpoint(model, checkpoint_path):
    """Load a checkpoint containing only trainable PAT-LLM parameters."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    parameters = ms.load_checkpoint(checkpoint_path)
    not_loaded, unused = ms.load_param_into_net(model, parameters)
    trainable_names = {parameter.name for parameter in model.trainable_params()}
    missing_trainable = [
        name for name in not_loaded if name in trainable_names
    ]
    if missing_trainable or unused:
        raise RuntimeError(
            "checkpoint mismatch: missing trainable parameters="
            f"{missing_trainable}, unused parameters={unused}"
        )


def train_model(config, model, train_dataset, validation_dataset):
    """Train PAT-LLM and save the best validation checkpoint."""
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(
        model.trainable_params(), learning_rate=config.learning_rate
    )

    def forward_fn(batch_x, batch_y):
        prediction, target = select_target(
            config, model(batch_x), batch_y
        )
        return loss_fn(prediction, target), prediction

    gradient_fn = ms.value_and_grad(
        forward_fn,
        grad_position=None,
        weights=optimizer.parameters,
        has_aux=True,
    )
    best_loss = None
    best_epoch = 0
    stale_epochs = 0
    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, config.train_epochs + 1):
        model.set_train(True)
        epoch_start = time.time()
        train_losses = []
        for batch_x, batch_y, _, _ in train_dataset.create_tuple_iterator(
            num_epochs=1
        ):
            (loss, _), gradients = gradient_fn(batch_x, batch_y)
            optimizer(gradients)
            train_losses.append(float(loss.asnumpy()))

        validation_loss, validation_mae = evaluate_loss(
            config, model, validation_dataset
        )
        train_loss = float(np.mean(train_losses))
        improved = best_loss is None or validation_loss < best_loss
        if improved:
            ms.save_checkpoint(
                model.trainable_params(), config.checkpoint_path
            )
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.7f} "
            f"val_mse={validation_loss:.7f} "
            f"val_mae={validation_mae:.7f} "
            f"seconds={time.time() - epoch_start:.1f}"
        )
        if stale_epochs >= config.patience:
            print(f"early stopping at epoch {epoch}")
            break
    return best_epoch


def save_evaluation(config, predictions, targets, histories):
    """Write reproducible numeric outputs without plotting dependencies."""
    os.makedirs(config.output_dir, exist_ok=True)
    metrics = calculate_metrics(predictions, targets)
    with open(
        os.path.join(config.output_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(metrics, stream, indent=2)
    np.savez_compressed(
        os.path.join(config.output_dir, "predictions.npz"),
        predictions=predictions,
        targets=targets,
        histories=histories,
    )
    print(json.dumps(metrics, indent=2))
    return metrics
