import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class CurveHead(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CurveFormerHead(nn.Module):
    """
    A simple decoder for quadratic curves.
    """
    def __init__(self,
                 in_channels=[32, 64, 160, 256],
                 hidden_dim=2048,
                 num_queries=400,
                 num_classes=2,
                 embedding_dim=256,
                 dropout_ratio=0.1,
                 feature_map_size=(300, 200)):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.feature_map_size = feature_map_size

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.conv = nn.Conv2d(embedding_dim*4, num_queries, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(num_queries)
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.curve_embed = CurveHead(self.feature_map_size[0] * self.feature_map_size[1], hidden_dim, 8, 3)
        self.class_embed = nn.Linear(self.feature_map_size[0] * self.feature_map_size[1], num_classes+1)

    def forward(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = self.conv(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.bn(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = x.flatten(2)
        output_curves = self.curve_embed(x).sigmoid()
        output_class = self.class_embed(x)

        return {'pred_logits': output_class, 'pred_curves': output_curves}
