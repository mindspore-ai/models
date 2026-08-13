"""Train PAT-LLM with MindSpore."""

from src import PATLLM, create_dataset, load_config
from src.engine import (
    attach_dataset_metadata,
    configure_runtime,
    load_trainable_checkpoint,
    predict,
    save_evaluation,
    train_model,
)


def main():
    """Run standalone training followed by test-set evaluation."""
    config = load_config("Train PAT-LLM with MindSpore")
    configure_runtime(config)
    train_source, train_dataset = create_dataset(config, "train")
    _, validation_dataset = create_dataset(config, "val")
    _, test_dataset = create_dataset(config, "test")
    attach_dataset_metadata(config, train_source)
    model = PATLLM(config)
    best_epoch = train_model(
        config, model, train_dataset, validation_dataset
    )
    print(f"best validation checkpoint: epoch {best_epoch}")
    load_trainable_checkpoint(model, config.checkpoint_path)
    predictions, targets, histories = predict(config, model, test_dataset)
    save_evaluation(config, predictions, targets, histories)


if __name__ == "__main__":
    main()
