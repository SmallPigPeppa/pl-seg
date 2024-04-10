from fastai.vision.all import *
path = untar_data(URLs.CAMVID_TINY)
path = untar_data(URLs.PASCAL_2007)
path.ls()


codes = np.loadtxt(path/'codes.txt', dtype=str)
print(codes)

# # Image localization datasets
# BIWI_HEAD_POSE = f"{S3_IMAGELOC}biwi_head_pose.tgz"
# CAMVID = f'{S3_IMAGELOC}camvid.tgz'
# CAMVID_TINY = f'{URL}camvid_tiny.tgz'
# LSUN_BEDROOMS = f'{S3_IMAGE}bedroom.tgz'
# PASCAL_2007 = f'{S3_IMAGELOC}pascal_2007.tgz'
# PASCAL_2012 = f'{S3_IMAGELOC}pascal_2012.tgz'

import wandb

def label_func(fn):
    return fn.parent.parent / "labels" / f"{fn.stem}_P{fn.suffix}"

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
    # artifact = wandb.use_artifact(artifact_id, type="dataset")
    # artifact_dir = Path(artifact.download())

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
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image_size = (image_shape[0] // resize_factor, image_shape[1] // resize_factor)
    dls = SegmentationDataLoaders.from_label_func(
        artifact_dir,
        bs=batch_size,
        fnames=fnames,
        label_func=label_func,
        codes=codes,
        batch_tfms=[*aug_transforms(size=image_size),
                    Normalize.from_stats(*imagenet_stats)],
        valid_pct=validation_split,
        seed=seed)

    return dls, class_labels
