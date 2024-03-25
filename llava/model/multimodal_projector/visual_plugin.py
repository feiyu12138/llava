import torch
import torch.nn as nn
from functools import partial
from timm.models.regnet import RegStage
from timm.layers import LayerNorm2d
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
)

def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Abstractor(nn.Module):
    def __init__(self, hidden_dim, num_pre_layers, num_post_layers, pool_stride, grouping):
        super(Abstractor, self).__init__()
        self.type = grouping.split('_')[0] # option: cabstractor, dabstractor
        self.is_gate = grouping.find('gate')!=-1
        
        if self.type == 'cabstractor':
            RegBlock = partial(
                RegStage,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
            s1 = RegBlock(
                num_pre_layers,
                hidden_dim,
                hidden_dim,
            )
            s2 = RegBlock(
                num_post_layers,
                hidden_dim,
                hidden_dim,
            )
            sampler = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
            self.net = nn.Sequential(s1, sampler, s2)
        elif self.type == 'dabstractor':
            self.net = nn.Identity()
        elif self.type == 'DWConvabstractor':
            depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=pool_stride+1, stride=pool_stride, padding=pool_stride//2, groups=hidden_dim, bias=False)
            norm = LayerNorm2d(hidden_dim)
            act = nn.SiLU()
            self.net = nn.Sequential(depthwise, norm, act)
        else:
            self.net = nn.Identity()

        if self.is_gate:
            self.pooler = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
            self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self,x):
        if self.is_gate:
            x = self.net(x) * self.gate.tanh() + self.pooler(x)
        else:
            x = self.net(x)
        return x

class DAbstractor(nn.Module):
    def __init__(self,config, num_feature_levels,decoder_layers ):
        super(Abstractor, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(config) for _ in range(decoder_layers)]
        )
        
    
if __name__ == '__main__':
    model = Abstractor(32, 3, 3, 4, 'DWConvabstractor_gate')
    x = torch.randn(1, 32, 24, 24)
    y = model(x)
    print(y.shape)
        