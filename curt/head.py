import copy
import numpy as np
import torch.nn.functional as F
import torch

from typing import Optional, List, Tuple

from torch import Tensor, nn

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self,
                 input_dim: int = 2048,
                 embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class CurveHead(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int):
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
                 in_channels: Tuple[int, int, int, int] = (32, 64, 160, 256),
                 num_queries: int = 400,
                 num_classes: int = 1,
                 embedding_dim: int = 256,
                 dropout: float = 0.1,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 activation: str = 'relu',
                 normalize_before: bool = False,
                 num_decoder_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_heads = nhead

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.conv = nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.dropout = nn.Dropout2d(dropout)

        decoder_layer = TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        decoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.query_embed = nn.Embedding(num_queries, embedding_dim)

        self.curve_embed = CurveHead(embedding_dim, embedding_dim, 8, 3)
        self.class_embed = nn.Linear(embedding_dim, num_classes+1)

    def forward(self,
                features: Tuple[Tensor, Tensor, Tensor, Tensor],
                mask) -> Tuple[Tensor, Tensor]:
        c1, c2, c3, c4 = features

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
        features = F.relu(x)

        x = self.dropout(features)
        x = x.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, n, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(tgt, x, memory_key_padding_mask=mask, query_pos=query_embed)

        output_curves = self.curve_embed(hs).sigmoid()
        output_class = self.class_embed(hs)

        return output_class, output_curves


class SegmentationHead(nn.Module):
    """
    A combined bezier curve and segmentation prediction head.
    """
    def __init__(self,
                 curve_head: CurveFormerHead):
        super().__init__()

        self.curve_head = curve_head

        for p in self.curve_head.parameters():
            p.requires_grad_(False)

        hidden_dim = self.curve_head.embedding_dim
        num_heads = self.curve_head.num_heads
        self.curve_attention = MHAttentionMap(hidden_dim, hidden_dim, num_heads)
        self.mask_head = MaskHeadSmallConv(hidden_dim,  num_heads, 32)

    def forward(self,
                features: Tuple[Tensor, Tensor, Tensor, Tensor],
                mask):
        c1, c2, c3, c4 = features

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.curve_head.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.curve_head.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.curve_head.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.curve_head.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = self.curve_head.conv(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.curve_head.bn(x)
        features = F.relu(x)

        x = self.curve_head.dropout(features)
        x = x.flatten(2).permute(2, 0, 1)
        query_embed = self.curve_head.query_embed.weight.unsqueeze(1).repeat(1, n, 1)
        tgt = torch.zeros_like(query_embed)

        hs = self.curve_head.decoder(tgt, x, memory_key_padding_mask=mask.flatten(1), query_pos=query_embed)

        # predict curves and classes
        output_curves = self.curve_head.curve_embed(hs).sigmoid()
        output_class = self.curve_head.class_embed(hs)

        curve_mask = self.curve_attention(hs[-1], features, mask=mask)

        # features: torch.Size([1, 256, 194, 152])
        # curve_mask: torch.Size([1, 200, 8, 194, 152])
        # seg_masks: torch.Size([200, 1, 194, 152])
        # outputs_seg_masks: torch.Size([1, 200, 194, 152])
        seg_masks = self.mask_head(features, curve_mask)
        outputs_seg_masks = seg_masks.view(n, self.curve_head.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        return {'pred_logits': output_class[-1], 'pred_curves': output_curves[-1], 'pred_masks': outputs_seg_masks}


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head with a bottleneck layer.
    """
    def __init__(self, feature_dim: int, att_dim: int, inter_dim: int):
        super().__init__()
        self.bottle_conv = torch.nn.Conv2d(feature_dim, inter_dim, 1)
        self.gn1 = torch.nn.GroupNorm(8, inter_dim)
        self.conv_1 = torch.nn.Conv2d(inter_dim + att_dim, inter_dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dim)
        self.conv_2 = torch.nn.Conv2d(inter_dim, inter_dim, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dim)
        self.out_lay = torch.nn.Conv2d(inter_dim, 1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Tensor, curve_mask: Tensor):
        x = self.bottle_conv(inputs)
        x = torch.cat([_expand(x, curve_mask.shape[1]), curve_mask.flatten(0, 1)], 1)
        x = self.conv_1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.gn2(x)
        x = F.relu(x)
        return self.out_lay(x)


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MHAttentionMap(nn.Module):
    """
    This is a 2D attention module, which only returns the attention softmax (no
    multiplication by value)
    """

    def __init__(self, query_dim: int, hidden_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        return weights


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers: int, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0).transpose(1, 2)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

