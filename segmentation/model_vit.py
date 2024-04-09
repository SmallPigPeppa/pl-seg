import torch
from einops import rearrange
import math
from timm import create_model
from timm.models._manipulate import checkpoint_seq
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
import torch.nn.functional as F


class Encoder2D(nn.Module):
    def __init__(self, decode_depth, hidden_dim, image_size, patch_size, backbone):
        super().__init__()
        self.backbone = backbone
        self.decode_depth = decode_depth
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.num_ps = self.calculate_num_ps()
        self.decode_scale = self.calculate_decode_scale()
        self.fc = nn.Linear(hidden_dim, hidden_dim * self.decode_scale[0] * self.decode_scale[1])

        self.weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
        self.net = create_model(self.backbone, pretrained=True, dynamic_img_size=True)
        self.modified()

    def calculate_num_ps(self):
        # Calculate the number of patches along each dimension
        assert self.image_size[0] % self.patch_size[0] == 0, "Image height must be divisible by patch height."
        assert self.image_size[1] % self.patch_size[1] == 0, "Image width must be divisible by patch width."
        n_ps0 = self.image_size[0] // self.patch_size[0]
        n_ps1 = self.image_size[1] // self.patch_size[1]
        return (n_ps0, n_ps1)

    def calculate_decode_scale(self):
        # Calculate the number of patches along each dimension
        k = int(math.pow(2, self.decode_depth))
        assert self.patch_size[0] % k == 0, f"Patch height must be divisible by {k}."
        assert self.patch_size[1] % k == 0, f"Patch width must be divisible by {k}."
        decode_scale0 = self.patch_size[0] // k
        decode_scale1 = self.patch_size[1] // k
        return (decode_scale0, decode_scale1)

    def forward(self, x):
        # vit forward
        patch_embed_x = self.patch_embed(x, patch_size=self.patch_size)
        encode_x = self.forward_features(patch_embed_x)[:, 1:, :]

        # fc for decode
        x = self.fc(encode_x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=self.decode_scale[0],
                      p2=self.decode_scale[1], h=self.num_ps[0], w=self.num_ps[1],
                      c=self.hidden_dim)

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

    def modified(self):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        if self.net.dynamic_img_size:
            # flatten deferred until after pos embed
            self.embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = self.get_patch_embed(image_size=224, patch_size=(16, 16))
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
                patch_embed=origin_weight, new_patch_size=patch_size
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias = nn.Parameter(self.net.patch_embed.proj.bias.clone().detach(),
                                                     requires_grad=True)

        return new_patch_embed


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
                 vit_image_size=(224, 224),
                 patch_size=(16, 16),
                 num_classes=21,
                 hidden_dim=768,
                 decode_features=[512, 256, 128, 64],
                 backbone='vit_base_patch16_224'):
        super().__init__()
        self.vit_image_size = vit_image_size
        self.encoder_2d = Encoder2D(
            decode_depth=len(decode_features),
            hidden_dim=hidden_dim,
            image_size=vit_image_size,
            patch_size=patch_size,
            backbone=backbone)
        self.decoder_2d = Decoder2D(
            in_channels=hidden_dim,
            out_channels=num_classes,
            features=decode_features)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.interpolate(x, size=self.vit_image_size, mode="bilinear", align_corners=False)
        _, final_x = self.encoder_2d(x)
        x = self.decoder_2d(final_x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    # def __init__(self, backbone="mobilenetv2_100", hidden_dim=256, num_classes=21):
    net = SegmentationModel(
        vit_image_size=(224, 224),
        patch_size=(16, 16),
        num_classes=21,
        hidden_dim=768,
        decode_features=[512, 256, 128, 64],
        backbone='vit_base_patch16_224')
    t1 = torch.rand(16, 3, 224, 224)
    print("input: " + str(t1.shape))

    # print(net)
    print("output: " + str(net(t1).shape))

    net = SegmentationModel(
        vit_image_size=(224, 224),
        patch_size=(16, 16),
        num_classes=21,
        hidden_dim=768,
        decode_features=[512, 256, 128, 64],
        backbone='vit_base_patch16_224')
    t1 = torch.rand(16, 3, 180, 240)
    print("input: " + str(t1.shape))

    # print(net)
    print("output: " + str(net(t1).shape))
