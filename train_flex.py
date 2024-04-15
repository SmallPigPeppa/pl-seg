import gc
import wandb
import ml_collections
import torch
from fastai.torch_core import set_seed
from fastai.metrics import DiceMulti, foreground_acc, JaccardCoeffMulti, APScoreMulti

from configs import get_config
from segmentation.camvid_utils import get_dataloader
from segmentation.train_utils import get_learner_flex as get_learner


def train_fn(configs: ml_collections.ConfigDict):
    wandb_configs = configs.wandb_configs
    experiment_configs = configs.experiment_configs
    loss_alias_mappings = configs.loss_mappings
    inference_config = configs.inference

    run = wandb.init(
        name=wandb_configs.name,
        project=wandb_configs.project,
        entity=wandb_configs.entity,
        job_type=wandb_configs.job_type,
        config=experiment_configs.to_dict(),
    )
    set_seed(wandb.config.seed)

    data_loader, class_labels = get_dataloader(
        artifact_id=configs.wandb_configs.artifact_id,
        batch_size=wandb.config.batch_size,
        image_shape=(wandb.config.image_height, wandb.config.image_width),
        resize_factor=wandb.config.image_resize_factor,
        validation_split=wandb.config.validation_split,
        seed=wandb.config.seed,
    )

    learner = get_learner(
        image_shape=(wandb.config.image_height, wandb.config.image_width),
        resize_factor=wandb.config.image_resize_factor,
        data_loader=data_loader,
        backbone=wandb.config.backbone,
        hidden_dim=wandb.config.hidden_dims,
        num_classes=len(class_labels),
        checkpoint_file=None,
        loss_func=loss_alias_mappings[wandb.config.loss_function](axis=1),
        metrics=[foreground_acc, JaccardCoeffMulti(), DiceMulti()],
        log_preds=False,
    )

    if wandb.config.fit == "fit":
        learner.fit_one_cycle(
            wandb.config.num_epochs,
            wandb.config.learning_rate,
            wd=wandb.config.weight_decay,
        )
    else:
        learner.fine_tune(
            wandb.config.num_epochs,
            wandb.config.learning_rate,
            wd=wandb.config.weight_decay,
        )
    del learner



if __name__ == "__main__":
    cfg = get_config()
    train_fn(cfg)
