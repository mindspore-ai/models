"""YAML configuration loader for PAT-LLM entry points."""

import argparse
from types import SimpleNamespace

import yaml


def _parse_scalar(value):
    """Parse an override value with YAML scalar semantics."""
    parsed = yaml.safe_load(value)
    if isinstance(parsed, (dict, list)):
        raise ValueError("configuration overrides must be scalar values")
    return parsed


def _build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config_path",
        type=str,
        default="default_config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one YAML value; this option can be repeated.",
    )
    return parser


def load_config(description="PAT-LLM"):
    """Load YAML and apply command-line ``--set KEY=VALUE`` overrides."""
    arguments = _build_parser(description).parse_args()
    with open(arguments.config_path, "r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError("the top level of the configuration must be a mapping")

    for item in arguments.overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}; expected KEY=VALUE")
        key, raw_value = item.split("=", 1)
        if key not in values:
            raise KeyError(f"unknown configuration key: {key}")
        values[key] = _parse_scalar(raw_value)

    _validate(values)
    return SimpleNamespace(**values)


def _validate(values):
    required = {
        "aap_recent_fraction",
        "aap_top_k",
        "aap_top_m",
        "allow_download",
        "batch_size",
        "checkpoint_path",
        "d_ff",
        "device_target",
        "device_id",
        "description",
        "dropout",
        "features",
        "freq",
        "label_len",
        "learning_rate",
        "llm_dtype",
        "llm_layers",
        "root_path",
        "data_path",
        "target",
        "seq_len",
        "pred_len",
        "d_model",
        "n_heads",
        "llm_model_path",
        "num_prototypes",
        "num_workers",
        "output_dir",
        "patch_len",
        "patience",
        "percent",
        "seed",
        "stride",
        "train_epochs",
        "trust_remote_code",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise KeyError(f"missing required configuration values: {missing}")
    if values["device_target"] not in {"CPU", "GPU", "Ascend"}:
        raise ValueError("device_target must be CPU, GPU, or Ascend")
    if values.get("features", "MS") not in {"M", "MS", "S"}:
        raise ValueError("features must be M, MS, or S")
    positive_values = (
        "seq_len",
        "label_len",
        "pred_len",
        "patch_len",
        "stride",
        "d_model",
        "n_heads",
        "d_ff",
        "batch_size",
        "train_epochs",
        "llm_layers",
        "num_prototypes",
        "aap_top_k",
        "aap_top_m",
        "patience",
    )
    for name in positive_values:
        if int(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if values["label_len"] > values["seq_len"]:
        raise ValueError("label_len cannot exceed seq_len")
    if values["patch_len"] > values["seq_len"] + values["stride"]:
        raise ValueError("patch_len exceeds the padded sequence length")
    if not 1 <= int(values.get("percent", 100)) <= 100:
        raise ValueError("percent must be between 1 and 100")
    if float(values["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= float(values["dropout"]) < 1:
        raise ValueError("dropout must be in the interval [0, 1)")
    if not 0 < float(values["aap_recent_fraction"]) <= 1:
        raise ValueError("aap_recent_fraction must be in the interval (0, 1]")
    if values.get("llm_dtype", "float16") not in {
        "float16",
        "bfloat16",
        "float32",
    }:
        raise ValueError("llm_dtype must be float16, bfloat16, or float32")
