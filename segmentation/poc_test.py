from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)

# 过滤函数，用于确定文件是否有对应的分割掩码
def has_segmentation_mask(f):
    return (path / f'segmentation/{f.stem}.png').exists()

# 定义如何从文件名获取标签的函数
def get_y_fn(x):
    return Path(str(x).replace("images", "segmentation").replace(".jpg", ".png"))

# 自定义获取图像文件的函数，同时应用过滤条件
def get_image_files_filtered(path):
    return [f for f in get_image_files(path) if has_segmentation_mask(f)]

image_size = (224, 224)
# 使用DataBlock定义数据
segmentation_block = DataBlock(
    blocks=(ImageBlock, MaskBlock), # 定义输入和输出的类型
    get_items=get_image_files_filtered, # 使用过滤后的图像文件获取函数
    splitter=FuncSplitter(lambda x: Path(x).parent.parent.name == 'test'), # 分割训练集和测试集
    get_y=get_y_fn, # 定义如何获取标签
    item_tfms=Resize(image_size), # 应用图像尺寸调整
    # batch_tfms=[*aug_transforms(size=image_size)] # 应用增强变换
)

# 根据路径加载数据
dls = segmentation_block.dataloaders(path, path={"train": path/"train", "valid": path/"test"}, bs=8)

# 展示一部分数据
dls.show_batch(max_n=4)
