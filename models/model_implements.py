import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.Swin import SwinTransformer
from models.heads.UPerHead import M_UPerHead

from collections import OrderedDict


class Swin_T(nn.Module):
    def __init__(self, num_classes=2, in_channel=3):
        super(Swin_T, self).__init__()

        self.swin_transformer = SwinTransformer(in_chans=in_channel,
                                                embed_dim=96,
                                                depths=[2, 2, 6, 2],
                                                num_heads=[3, 6, 12, 24],
                                                window_size=7,
                                                mlp_ratio=4.,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=0.,
                                                attn_drop_rate=0.,
                                                drop_path_rate=0.3,
                                                ape=False,
                                                patch_norm=True,
                                                out_indices=(0, 1, 2, 3),
                                                use_checkpoint=False)

        self.uper_head = M_UPerHead(in_channels=[96, 192, 384, 768],
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_classes=num_classes,
                                    align_corners=False,)

    def load_pretrained_imagenet(self, dst, device):
        pretrained_states = torch.load(dst)['model']
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if 'head.weight' == item or 'head.bias' == item or 'norm.weight' == item or 'norm.bias' == item or 'layers.0.blocks.1.attn_mask' == item or 'layers.1.blocks.1.attn_mask' == item or 'layers.2.blocks.1.attn_mask' == item or 'layers.2.blocks.3.attn_mask' == item or 'layers.2.blocks.5.attn_mask' == item:
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        # temporally remove fpn norm layers that not included on official model
        self.swin_transformer.remove_fpn_norm_layers()
        self.swin_transformer.load_state_dict(pretrained_states_backbone)
        self.swin_transformer.add_fpn_norm_layers()
        self.to(device)

    def forward(self, x):
        x_size = x.shape[2:]

        feat1, feat2, feat3, feat4 = self.swin_transformer(x)
        feat = self.uper_head(feat1, feat2, feat3, feat4)
        feat = F.interpolate(feat, x_size, mode='bilinear', align_corners=False)

        return feat
