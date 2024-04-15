import wandb
from absl import app, flags

from ml_collections.config_flags import config_flags

from train import train_fn

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    config = FLAGS.experiment_configs

    # 使用你已经确定的最优参数
    best_params = {
        "batch_size": 8,
        "num_epochs": 90,
        "image_resize_factor": 2,
        "backbone": "resnet50",
        "loss_function": "categorical_cross_entropy",
        "learning_rate": 0.0001813,
        "weight_decay": 0.0425,
    }

    # 初始化 WandB
    wandb.init(
        project=config.wandb_configs.project,
        entity=config.wandb_configs.entity,
        config=best_params
    )

    # 运行训练函数，将最优参数作为配置传递
    train_fn(config=config, **best_params)

    # 当训练完成后，结束 WandB session
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
