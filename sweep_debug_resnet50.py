import wandb
from absl import app, flags

from functools import partial

from ml_collections.config_flags import config_flags

# from train import train_fn
from train_resnet50 import train_fn

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    config = FLAGS.experiment_configs
    sweep_configs = {
        "method": config.sweep_method,
        "metric": {"name": config.sweep_metric_name, "goal": config.sweep_goal},
        "early_terminate": {
            "type": config.early_terminate_type,
            "min_iter": config.early_terminate_min_iter,
        },
        "parameters": {
            "batch_size": {"values": [4, 8, 16]},
            "num_epochs": {"values": [5, 10, 20, 40]},
            "image_resize_factor": {"values": [2, 4]},
            "backbone": {
                "values": [
                    "resnet50",
                    # "vit-b",
                ]
            },
            "loss_function": {"values": ["categorical_cross_entropy", "focal", "dice"]},
            "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
            "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.05},
        },
    }
    sweep_id = wandb.sweep(
        sweep_configs,
        project=config.wandb_configs.project,
        entity=config.wandb_configs.entity,
    )
    wandb.agent(sweep_id, function=partial(train_fn, config), count=config.sweep_count)


if __name__ == "__main__":
    app.run(main)
