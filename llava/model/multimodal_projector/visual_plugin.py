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


class CAbstractor(nn.Module):
    def __init__(self, hidden_dim, num_pre_layers, num_post_layers, pool_stride):
        super(CAbstractor, self).__init__()
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
        sampler = nn.AvgPool2d(kernel_size=pool_stride,stride=pool_stride)
        
        s2 = RegBlock(
            num_post_layers,
            hidden_dim,
            hidden_dim,
        )
        self.net = nn.Sequential(s1, sampler, s2)
        # self.readout = build_mlp(mlp_layers, hidden_dim, out_dim)
    
    def forward(self,x):
        x = self.net(x)
        return x

class DAbstractor(nn.Module):
    def __init__(self,config, num_feature_levels,decoder_layers ):
        super(CAbstractor, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(config) for _ in range(decoder_layers)]
        )
        
    
if __name__ == '__main__':
    model = CAbstractor(3, 3, 3, 3, 3, 3, 2)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
        