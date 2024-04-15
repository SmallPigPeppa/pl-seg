import torch
from segmentation.flexivit import ClassificationEvaluator
import math
import torch.nn as nn
import torch.nn.functional as F


class Encoder2D(nn.Module):
    def __init__(self, decode_depth, hidden_dim, image_size):
        super().__init__()
        self.decode_depth = decode_depth
        self.image_size = image_size
        self.patch_size, self.vit_image_size, self.num_patch = self.find_optimal_patch_and_size(
            image_size=self.image_size, depth=self.decode_depth
        )
        self.hidden_dim = hidden_dim

        args = {
            # 'checkpoint_path': '/mnt/mmtech01/usr/liuwenzhuo/code/test-code/flexivit/ckpt/L2P/add_random_resize_4conv_fix14token_2range_ratio/last.ckpt',
            'weights': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            'num_classes': 1000,
            'image_size': self.vit_image_size,
            'patch_size': self.patch_size,
            'resize_type': "pi"
        }
        # self.net = ClassificationEvaluator.load_from_checkpoint(strict=True, **args)
        self.net = ClassificationEvaluator(**args)

    def find_optimal_patch_and_size(self, image_size, depth, min_patches=12, max_patches=16):
        """
        根据输入的图像尺寸、深度和patch数量范围，找到最优的 patch size 和调整后的 image size。

        参数:
        image_size (tuple): 输入的图像宽高 (width, height)
        depth (int): 深度，patch size 必须是 2^depth 的整数倍
        min_patches (int): 分割后的最小块数
        max_patches (int): 分割后的最大块数

        返回:
        tuple: 最优的 patch size 和调整后的 image size
        """
        base = 2 ** depth
        best_patch_sizes = []
        best_adjusted_sizes = []
        best_num_patches = []

        for dimension in image_size:
            best_patch_diff = float('inf')
            best_patch_size = None
            best_adjusted_size = None
            best_patch_count = None

            # 遍历可能的patch size
            for patch_size in range(base, dimension + 1, base):
                num_patches = math.ceil(dimension / patch_size)
                if min_patches <= num_patches <= max_patches:
                    adjusted_size = patch_size * num_patches
                    patch_diff = abs(dimension - adjusted_size)

                    # 寻找变化最小的调整后尺寸
                    if patch_diff < best_patch_diff:
                        best_patch_diff = patch_diff
                        best_patch_size = patch_size
                        best_adjusted_size = adjusted_size
                        best_patch_count = num_patches

            best_patch_sizes.append(best_patch_size)
            best_adjusted_sizes.append(best_adjusted_size)
            best_num_patches.append(best_patch_count)

        return best_patch_sizes, best_adjusted_sizes, best_num_patches

        # return best_patch_sizes, best_adjusted_sizes

    def forward(self, x):
        B, C, H, W = x.shape
        # vit forward
        x = self.net.forward_seg(x)[:, 1:, :]

        # align feature B (H*W) (C*D) -> B (C*D) (H*W) -> B (C*D) H W
        xp = x.permute(0, 2, 1).reshape(B, -1, self.num_patch[0], self.num_patch[1])

        return xp


class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[3], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self,
                 image_size=(224, 224),
                 # num_patch=(16, 16),
                 # patch_size=(16, 16),
                 num_classes=21,
                 hidden_dim=768,
                 decode_features=[512, 256, 128, 64]):
        super().__init__()
        # self.vit_image_size = vit_image_size
        self.encoder_2d = Encoder2D(
            decode_depth=len(decode_features),
            hidden_dim=hidden_dim,
            image_size=image_size,
            # num_patch=num_patch
            # patch_size=patch_size,
            # backbone=backbone
        )
        self.decoder_2d = Decoder2D(
            in_channels=hidden_dim,
            out_channels=num_classes,
            features=decode_features)

    def forward(self, x):
        b, c, h, w = x.shape
        # x = F.interpolate(x, size=self.vit_image_size, mode="bilinear", align_corners=False)
        x = self.encoder_2d(x)
        x = self.decoder_2d(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    k = 8
    image_size = (720 // k, 960 // k)

    net = SegmentationModel(
        image_size=image_size,
        num_classes=21,
        hidden_dim=768,
        decode_features=[512, 256, 128, 64],
        # backbone='vit_base_patch16_224'
        )
    t1 = torch.rand(16, 3, *image_size)
    print("input: " + str(t1.shape))

    print("output: " + str(net(t1).shape))
