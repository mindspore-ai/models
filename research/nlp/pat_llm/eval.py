"""Evaluate a trained PAT-LLM MindSpore checkpoint."""

from src import PATLLM, create_dataset, load_config
from src.engine import (
    attach_dataset_metadata,
    configure_runtime,
    load_trainable_checkpoint,
    predict,
    save_evaluation,
)


def main():
    """Load a trainable-parameter checkpoint and evaluate it."""
    config = load_config("Evaluate PAT-LLM with MindSpore")
    configure_runtime(config)
    test_source, test_dataset = create_dataset(config, "test")
    attach_dataset_metadata(config, test_source)
    model = PATLLM(config)
    load_trainable_checkpoint(model, config.checkpoint_path)
    predictions, targets, histories = predict(config, model, test_dataset)
    save_evaluation(config, predictions, targets, histories)


if __name__ == "__main__":
    main()
