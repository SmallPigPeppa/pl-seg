import wandb
from typing import Tuple, Dict
# from fastai.vision.all import *
from fastai.vision.all import Image, Path, Resize, progress_bar, get_image_files, SegmentationDataLoaders,aug_transforms,Normalize,imagenet_stats
import numpy as np


def label_func(fn):
    return fn.parent.parent / "labels" / f"{fn.stem}_P{fn.suffix}"


def get_frequency_distribution(mask_data, class_labels):
    (unique, counts) = np.unique(mask_data, return_counts=True)
    unique = list(unique)
    counts = list(counts)
    frequency_dict = {}
    for _class in class_labels.keys():
        if _class in unique:
            frequency_dict[class_labels[_class]] = counts[unique.index(_class)]
        else:
            frequency_dict[class_labels[_class]] = 0
    return frequency_dict


def log_dataset(project: str, entity: str, artifact_id: str, class_labels: Dict):
    with wandb.init(
            project=project, name="visualize_camvid", entity=entity, job_type="data_viz"
    ):
        artifact = wandb.use_artifact(artifact_id, type="dataset")
        artifact_dir = artifact.download()

        table_data = []
        image_files = get_image_files(Path(artifact_dir) / "images")
        labels = [str(class_labels[_lab]) for _lab in list(class_labels)]

        print("Creating Table...")
        for image_file in progress_bar(image_files):
            image = np.array(Image.open(image_file))
            mask_data = np.array(Image.open(label_func(image_file)))
            frequency_distribution = get_frequency_distribution(mask_data)
            table_data.append(
                [
                    str(image_file.name),
                    wandb.Image(image),
                    wandb.Image(
                        image,
                        masks={
                            "predictions": {
                                "mask_data": mask_data,
                                "class_labels": class_labels,
                            }
                        },
                    ),
                ]
                + [frequency_distribution[_lab] for _lab in labels]
            )
        wandb.log(
            {
                "CamVid_Dataset": wandb.Table(
                    data=table_data,
                    columns=["File_Name", "Images", "Segmentation_Masks"] + labels,
                )
            }
        )


# def get_dataloader(
#         artifact_id: str,
#         batch_size: int,
#         image_shape: Tuple[int, int],
#         resize_factor: int,
#         validation_split: float,
#         seed: int,
# ):
#     """Grab an artifact and creating a Pytorch DataLoader"""
#     artifact = wandb.use_artifact(artifact_id, type="dataset")
#     artifact_dir = Path(artifact.download())
#     codes = np.loadtxt(artifact_dir / "codes.txt", dtype=str)
#     fnames = get_image_files(artifact_dir / "images")
#     class_labels = {k: v for k, v in enumerate(codes)}
#     return (
#         SegmentationDataLoaders.from_label_func(
#             artifact_dir,
#             bs=batch_size,
#             fnames=fnames,
#             label_func=label_func,
#             codes=codes,
#             item_tfms=Resize(
#                 (image_shape[0] // resize_factor, image_shape[1] // resize_factor)
#             ),
#             valid_pct=validation_split,
#             seed=seed,
#         ),
#         class_labels,
#     )


def get_dataloader(
        artifact_id: str,
        batch_size: int,
        image_shape: Tuple[int, int],
        resize_factor: int,
        validation_split: float,
        seed: int,
        normalize: bool = True,  # 新增参数控制是否进行归一化
):
    """Grab an artifact and creating a Pytorch DataLoader with data augmentation and normalization."""
    artifact = wandb.use_artifact(artifact_id, type="dataset")
    artifact_dir = Path(artifact.download())

    # 检查数据集是否已下载
    artifact_dir = Path('/mnt/mmtech01/usr/liuwenzhuo/torch_ds/pl-seg') / artifact_id  # 设置存储下载数据的路径
    if not artifact_dir.exists():  # 如果路径不存在，说明需要下载数据集
        artifact = wandb.use_artifact(artifact_id, type="dataset")
        artifact_dir = Path(artifact.download())
    else:
        print(f"Using cached dataset at {artifact_dir}")



    codes = np.loadtxt(artifact_dir / "codes.txt", dtype=str)
    fnames = get_image_files(artifact_dir / "images")
    class_labels = {k: v for k, v in enumerate(codes)}

    # 定义数据增强
    batch_tfms = [
        aug_transforms(mult=2.0, do_flip=True, flip_vert=True, max_rotate=45.0, max_zoom=1.1, max_lighting=0.2,
                       max_warp=0.2, p_affine=0.75, p_lighting=0.75, xtra_tfms=None, size=None, mode='bilinear',
                       pad_mode='reflection', align_corners=True, batch=False, min_scale=1.0)]
    if normalize:
        # 加入归一化，这里使用ImageNet的均值和标准差作为示例
        batch_tfms.append(Normalize.from_stats(*imagenet_stats))

    # 使用数据增强和归一化创建DataLoaders
    dls = SegmentationDataLoaders.from_label_func(
        artifact_dir,
        bs=batch_size,
        fnames=fnames,
        label_func=label_func,  # 确保你有定义label_func
        codes=codes,
        item_tfms=Resize((image_shape[0] // resize_factor, image_shape[1] // resize_factor)),
        batch_tfms=batch_tfms,
        valid_pct=validation_split,
        seed=seed,
    )
    return dls, class_labels