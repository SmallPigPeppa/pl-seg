from absl import app, flags
import wandb

from functools import partial

import ml_collections
from ml_collections.config_flags import config_flags

from train import train_fn

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")

def main(_):
    config = FLAGS.experiment_configs
    sweep_configs = {
        "method": config.sweep_method,
        "metric": {
            "name": config.sweep_metric_name,
            "goal": config.sweep_goal
        },
        "early_terminate": {
            "type": config.early_terminate_type,
            "min_iter": config.early_terminate_min_iter,
        },
        "parameters": {
            "batch_size": {"values": [4, 8, 16]},
            "image_resize_factor": {"values": [2, 4]},
            "backbone": {
                "values": [
                    "mobilenetv2_100",
                    "mobilenetv3_small_050",
                    "mobilenetv3_large_100",
                    "resnet18",
                    "resnet34",
                    "resnet50",
                    "vgg19",
                    "vgg16",
                ]
            },
            "loss_function": {"values": ["categorical_cross_entropy", "focal", "dice"]},
            "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
            "fit": {"values": ["fit", "fine-tune"]},
            "weight_decay":{"distribution": "uniform",
                            "min": 0.,
                            "max": 0.05}
        },
    }
    sweep_id = wandb.sweep(
        sweep_configs,
        project=config.wandb_configs.project,
        entity=config.wandb_configs.entity,
    )
    wandb.agent(
        sweep_id, function=partial(train_fn, config), count=config.sweep_count
    )


if __name__ == "__main__":
    app.run(main)
