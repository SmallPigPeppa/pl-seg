import os
from typing import Optional, Sequence, Tuple, Union
import torch
import pytorch_lightning as pl
import timm.models
from timm import create_model
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
from flexivit_pytorch.utils import resize_abs_pos_embed
from timm.models._manipulate import checkpoint_seq

from timm.layers import PatchEmbed
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
from flexivit_pytorch.utils import to_2tuple
import random


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            weights: str,
            num_classes: int,
            image_size: Union[int, Tuple[int, int]] = 240,
            patch_size: Union[int, Tuple[int, int]] = 14,
            resize_type: str = "pi",
    ):
        """Classification Evaluator

        Args:
            weights: Name of model weights
            n_classes: Number of target class.
            image_size: Size of input images
            patch_size: Resized patch size
            resize_type: Patch embed resize method. One of ["pi", "interpolate"]
            results_path: Path to write evaluation results. Does not write results if empty
        """
        super().__init__()
        self.weights = weights
        self.num_classes = num_classes
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.resize_type = resize_type

        # Load original weights
        print(f"Loading weights {self.weights}")
        self.net = create_model(self.weights, pretrained=True, dynamic_img_size=True)
        # modified
        self.modified(self.image_size, self.patch_size)

    def forward_features_after_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.forward_features(x)
        # x = self.net.forward_head(x)
        return x

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        # img_size_0 = self.num_patch[0] * self.patch_size[0]
        # img_size_1 = self.num_patch[1] * self.patch_size[1]
        x = F.interpolate(x, size=self.image_size, mode='bilinear')
        x = self.patch_embed_seg(x, patch_size=[self.patch_size[0], self.patch_size[1]])
        x = self.forward_features_after_patch_embed(x)

        return x

    def modified(self, image_size, patch_size):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        if self.net.dynamic_img_size:
            # flatten deferred until after pos embed
            self.embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        # self.patch_embed_4x4 = self.get_new_patch_embed(new_image_size=56, new_patch_size=4)
        # self.patch_embed_8x8 = self.get_new_patch_embed(new_image_size=112, new_patch_size=8)
        # self.patch_embed_12x12 = self.get_new_patch_embed(new_image_size=168, new_patch_size=12)
        # self.patch_embed_16x16 = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        # self.patch_embed_16x16_origin = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        self.patch_embed_seg = self.get_new_patch_embed(new_image_size=image_size, new_patch_size=patch_size)

        self.net.patch_embed = nn.Identity()

    def get_new_patch_embed(self, new_image_size, new_patch_size):
        new_patch_embed = FlexiPatchEmbed(
            img_size=new_image_size,
            patch_size=new_patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            # dynamic_img_pad=self.dynamic_img_pad,
            **self.embed_args,
        )
        if hasattr(self.net.patch_embed.proj, 'weight'):
            origin_weight = self.net.patch_embed.proj.weight.clone().detach()
            new_weight = pi_resize_patch_embed(
                patch_embed=origin_weight, new_patch_size=to_2tuple(new_patch_size)
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias = nn.Parameter(self.net.patch_embed.proj.bias.clone().detach(),
                                                     requires_grad=True)

        return new_patch_embed


if __name__ == "__main__":
    pass
