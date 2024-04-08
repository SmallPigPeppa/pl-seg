import logging
import math
import os
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from transformer_model import TransModel2d, TransConfig
import math

import os
import timm.models
from timm import create_model
from timm.models._manipulate import checkpoint_seq
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)


class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % (sample_v ** 2) == 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size,
                                     config.patch_size[0] * config.patch_size[1] * config.hidden_size // (
                                             sample_v ** 2))
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v

        self.is_segmentation = is_segmentation

        self.net = create_model(self.weights, pretrained=True)

    def forward(self, x):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]

        if h % p1 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        if w % p2 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        hh = h // p1
        ww = w // p2

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)
        a = self.bert_model(x)
        encode_x = self.bert_model(x)[-1]  # 取出来最后一层
        if not self.is_segmentation:
            return encode_x

        x = self.final_dense(encode_x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=self.hh, p2=self.ww, h=hh, w=ww,
                      c=self.config.hidden_size)
        return encode_x, x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.patch_embed(x)
        x = self.net._pos_embed(x)
        x = self.net.patch_drop(x)
        x = self.net.norm_pre(x)
        if self.net.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.net.blocks, x)
        else:
            x = self.net.blocks(x)
        x = self.net.norm(x)
        return x

    def modified(self, new_image_size=224, new_patch_size=16):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        if self.net.dynamic_img_size:
            # flatten deferred until after pos embed
            self.embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed_4x4 = self.get_patch_embed(image_size=56, patch_size=4)
        self.patch_embed_8x8 = self.get_patch_embed(image_size=112, patch_size=8)
        self.patch_embed_12x12 = self.get_patch_embed(image_size=168, patch_size=12)
        self.patch_embed_16x16 = self.get_patch_embed(image_size=224, patch_size=16)
        self.patch_embed_16x16_origin = self.get_patch_embed(image_size=224, patch_size=16)
        self.patch_embed = self.get_patch_embed(image_size=224,
                                                patch_size=[self.configpatch_size[0], self.configpatch_size[1]])

        self.net.patch_embed = nn.Identity()

    def get_patch_embed(self, image_size, patch_size):
        new_patch_embed = FlexiPatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            **self.embed_args,
        )
        if hasattr(self.net.patch_embed.proj, 'weight'):
            origin_weight = self.net.patch_embed.proj.weight.clone().detach()
            new_weight = pi_resize_patch_embed(
                patch_embed=origin_weight, new_patch_size=(patch_size, patch_size)
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias = nn.Parameter(self.net.patch_embed.proj.bias.clone().detach(),
                                                     requires_grad=True)

        return new_patch_embed


class PreTrainModel(nn.Module):
    def __init__(self, patch_size,
                 in_channels,
                 out_class,
                 hidden_size=1024,
                 num_hidden_layers=8,
                 num_attention_heads=16,
                 decode_features=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(patch_size=patch_size,
                             in_channels=in_channels,
                             out_channels=0,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out


class Vit(nn.Module):
    def __init__(self, patch_size,
                 in_channels,
                 out_class,
                 hidden_size=1024,
                 num_hidden_layers=8,
                 num_attention_heads=16,
                 sample_rate=4,
                 ):
        super().__init__()
        config = TransConfig(patch_size=patch_size,
                             in_channels=in_channels,
                             out_channels=0,
                             sample_rate=sample_rate,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)

        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out


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

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x


class SETRModel(nn.Module):
    def __init__(self, patch_size=(32, 32),
                 in_channels=3,
                 out_channels=1,
                 hidden_size=1024,
                 num_hidden_layers=8,
                 num_attention_heads=16,
                 decode_features=[512, 256, 128, 64],
                 sample_rate=4, ):
        super().__init__()
        config = TransConfig(patch_size=patch_size,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             sample_rate=sample_rate,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config)
        self.decoder_2d = Decoder2D(in_channels=config.hidden_size, out_channels=config.out_channels,
                                    features=decode_features)

    def forward(self, x):
        _, final_x = self.encoder_2d(x)
        x = self.decoder_2d(final_x)
        return x


if __name__ == "__main__":
    net = SETRModel(patch_size=(16, 16),
                    in_channels=3,
                    out_channels=1,
                    hidden_size=784,
                    num_hidden_layers=8,
                    num_attention_heads=16,
                    decode_features=[512, 256, 128, 64])
    t1 = torch.rand(16, 3, 224, 224)
    print("input: " + str(t1.shape))

    # print(net)
    print("output: " + str(net(t1).shape))
