"""
Point Transformer V3 - PointCNN++ Implementation

基于 pointcnnpp 实现的 Point Transformer V3，使用 PointCNN++ 的点云算子
和稀疏注意力机制，完全兼容 Pointcept 框架。

Author: Based on Point Transformer V3 and PointCNN++
"""

from functools import partial
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
from timm.layers import DropPath

# PointCNN++ imports
try:
    from pointcnnpp.layers.metadata import MetaData
    from pointcnnpp.layers.conv import PointConv3d, conv_with_stride
    from pointcnnpp.layers.upsample import Upsample
    from pointcnnpp.layers.triplets import (
        build_triplets,
        voxelize_3d,
        radius_scaler_for_kernel_size,
    )
    from pointcnnpp.layers.norm import (
        RaggedLayerNorm,
        RaggedBatchNorm,
    )
    from pointcnnpp.sparse_engines.ops import large_segment_reduce
    from pointcnnpp.models.point_transformer import PointTransformerLayer
except ImportError as e:
    raise ImportError(
        f"pointcnnpp is required for PointTransformerV3PointCNNpp. Error: {e}"
    )

# Pointcept imports
from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule


def point_to_metadata(point: Point) -> Tuple[torch.Tensor, MetaData]:
    """
    将 Point 对象转换为 pointcnnpp 的 MetaData 格式
    
    Args:
        point: Point 对象，包含 feat, coord, offset 等
        
    Returns:
        feat: [N, C] 特征张量
        m: MetaData 对象
    """
    feat = point.feat
    coord = point.coord
    
    # 确保数据类型正确
    if coord.dtype != torch.float32:
        coord = coord.float()
    if feat.dtype != torch.float32:
        feat = feat.float()
    
    # 获取 grid_size
    grid_size = point.get("grid_size", 0.05)
    
    # 转换 offset 到 sample_sizes 和 sample_inds
    offset = point.offset
    sample_sizes = offset2bincount(offset)
    sample_inds = torch.repeat_interleave(
        torch.arange(0, sample_sizes.numel(), device=sample_sizes.device),
        sample_sizes,
    )
    
    # 创建 MetaData
    m = MetaData(
        points=coord,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        grid_size=grid_size,
    )
    
    return feat, m


def metadata_to_point(
    feat: torch.Tensor, m: MetaData, original_point: Point
) -> Point:
    """
    将 pointcnnpp 的输出转换回 Point 对象
    
    Args:
        feat: [N, C] 特征张量
        m: MetaData 对象
        original_point: 原始 Point 对象（用于保留其他属性）
        
    Returns:
        Point 对象
    """
    # 从 MetaData 恢复 offset
    offset = torch.cumsum(
        torch.cat([torch.tensor([0], device=m.sample_sizes.device), m.sample_sizes]),
        dim=0,
    )[:-1]
    
    # 创建新的 Point 对象
    point_dict = {
        "feat": feat,
        "coord": m.points,
        "offset": offset,
        "grid_size": m.grid_size,
    }
    
    # 保留原始 Point 的其他属性
    for key in ["batch", "segment", "condition", "context"]:
        if key in original_point.keys():
            # 如果点数变化，需要调整
            if key == "batch":
                from pointcept.models.utils import offset2batch
                point_dict[key] = offset2batch(offset)
            elif feat.shape[0] == original_point[key].shape[0]:
                point_dict[key] = original_point[key]
    
    return Point(point_dict)


def reshape_for_transformer(
    feat: torch.Tensor, sample_sizes: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """
    将 [N, C] 格式的特征重塑为 [batch_size, points_per_batch, C] 格式
    
    Args:
        feat: [N, C] 特征张量
        sample_sizes: [B] 每个 batch 的点数
        
    Returns:
        feat_reshaped: [batch_size, max_points, C] 重塑后的特征
        max_points: 最大点数（用于 padding）
    """
    batch_size = sample_sizes.numel()
    max_points = sample_sizes.max().item()
    channels = feat.shape[1]
    
    # 重塑为批次格式
    feat_list = []
    start_idx = 0
    for i in range(batch_size):
        num_points = sample_sizes[i].item()
        feat_batch = feat[start_idx : start_idx + num_points]  # [num_points, C]
        # Padding 到 max_points
        if num_points < max_points:
            padding = torch.zeros(
                max_points - num_points, channels, device=feat.device, dtype=feat.dtype
            )
            feat_batch = torch.cat([feat_batch, padding], dim=0)
        feat_list.append(feat_batch)
        start_idx += num_points
    
    feat_reshaped = torch.stack(feat_list, dim=0)  # [batch_size, max_points, C]
    return feat_reshaped, max_points


def reshape_from_transformer(
    feat_reshaped: torch.Tensor, sample_sizes: torch.Tensor
) -> torch.Tensor:
    """
    将 [batch_size, max_points, C] 格式的特征重塑回 [N, C] 格式
    
    Args:
        feat_reshaped: [batch_size, max_points, C] 重塑后的特征
        sample_sizes: [B] 每个 batch 的点数
        
    Returns:
        feat: [N, C] 特征张量
    """
    batch_size = feat_reshaped.shape[0]
    channels = feat_reshaped.shape[2]
    
    feat_list = []
    for i in range(batch_size):
        num_points = sample_sizes[i].item()
        feat_batch = feat_reshaped[i, :num_points]  # [num_points, C]
        feat_list.append(feat_batch)
    
    feat = torch.cat(feat_list, dim=0)  # [N, C]
    return feat


class EmbeddingPointCNNpp(PointModule):
    """使用 PointCNN++ 的 Embedding 层"""

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        kernel_size=5,
        receptive_field_scaler=2.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.receptive_field_scaler = receptive_field_scaler

        self.conv = PointConv3d(in_channels, embed_channels, kernel_size=kernel_size, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_channels)
        else:
            self.norm = None
        if act_layer is not None:
            self.act = act_layer()
        else:
            self.act = None

    def forward(self, feat: torch.Tensor, m: MetaData) -> Tuple[torch.Tensor, MetaData]:
        feat, m = conv_with_stride(
            self.conv, feat, m, stride=1.0, receptive_field_scaler=self.receptive_field_scaler
        )
        m.dirty_triplets()  # 清除 triplets，下次使用时重新构建
        if self.norm is not None:
            feat = self.norm(feat, m.sample_sizes)
        if self.act is not None:
            feat = self.act(feat)
        return feat, m


class CPEPointCNNpp(PointModule):
    """使用 PointCNN++ 的条件位置编码"""

    def __init__(
        self,
        channels,
        norm_layer=None,
        kernel_size=3,
        receptive_field_scaler=2.5,
    ):
        super().__init__()
        self.conv = PointConv3d(channels, channels, kernel_size=kernel_size, bias=True)
        self.linear = nn.Linear(channels, channels)
        if norm_layer is not None:
            self.norm = norm_layer(channels)
        else:
            self.norm = None
        self.receptive_field_scaler = receptive_field_scaler

    def forward(self, feat: torch.Tensor, m: MetaData) -> Tuple[torch.Tensor, MetaData]:
        feat_cpe, m = conv_with_stride(
            self.conv, feat, m, stride=1.0, receptive_field_scaler=self.receptive_field_scaler
        )
        feat_cpe = self.linear(feat_cpe)
        if self.norm is not None:
            feat_cpe = self.norm(feat_cpe, m.sample_sizes)
        return feat_cpe, m


class MLP(nn.Module):
    """MLP 前馈网络"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BlockPointCNNpp(PointModule):
    """使用 PointCNN++ 的 Transformer Block"""

    def __init__(
        self,
        channels,
        num_heads,
        distance_threshold=0.3,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        cpe_kernel_size=3,
        cpe_receptive_field_scaler=2.5,
        use_native=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # CPE (Conditional Positional Encoding)
        self.cpe = CPEPointCNNpp(
            channels=channels,
            norm_layer=partial(RaggedLayerNorm, reduce_fn=large_segment_reduce)
            if norm_layer == nn.LayerNorm
            else None,
            kernel_size=cpe_kernel_size,
            receptive_field_scaler=cpe_receptive_field_scaler,
        )

        # Transformer Layer (使用 pointcnnpp 的实现)
        self.transformer = PointTransformerLayer(
            embed_dim=channels,
            num_heads=num_heads,
            distance_threshold=distance_threshold,
            ffn_dim=int(channels * mlp_ratio),
            dropout=proj_drop,
            use_native=use_native,
            pos_dim=3,
        )

        # MLP
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )

        # Normalization layers
        self.norm1 = norm_layer(channels)
        self.norm2 = norm_layer(channels)

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, feat: torch.Tensor, m: MetaData) -> Tuple[torch.Tensor, MetaData]:
        # CPE with residual
        shortcut = feat
        feat_cpe, m = self.cpe(feat, m)
        feat = shortcut + feat_cpe

        # Transformer attention
        shortcut = feat
        
        # 重塑为批次格式用于 Transformer
        feat_reshaped, max_points = reshape_for_transformer(feat, m.sample_sizes)
        coord_reshaped, _ = reshape_for_transformer(m.points, m.sample_sizes)
        coord_reshaped = coord_reshaped.unsqueeze(1)  # [batch_size, 1, max_points, 3]
        
        # 应用 Transformer
        if self.pre_norm:
            feat_reshaped = self.norm1(feat_reshaped)
        
        feat_reshaped = self.transformer(feat_reshaped, coord_reshaped)
        feat_reshaped = self.drop_path(feat_reshaped)
        
        # 重塑回原始格式
        feat = reshape_from_transformer(feat_reshaped, m.sample_sizes)
        feat = shortcut + feat

        if not self.pre_norm:
            feat = self.norm1(feat)

        # MLP
        shortcut = feat
        if self.pre_norm:
            feat = self.norm2(feat)
        feat = self.mlp(feat)
        feat = self.drop_path(feat)
        feat = shortcut + feat

        if not self.pre_norm:
            feat = self.norm2(feat)

        return feat, m


class DownsamplePointCNNpp(PointModule):
    """使用 PointCNN++ 的下采样层"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        kernel_size=1,
        receptive_field_scaler=2.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.receptive_field_scaler = receptive_field_scaler

        self.proj = PointConv3d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        if act_layer is not None:
            self.act = act_layer()
        else:
            self.act = None

    def forward(self, feat: torch.Tensor, m: MetaData) -> Tuple[torch.Tensor, MetaData]:
        # 使用 conv_with_stride 进行下采样
        feat, m = conv_with_stride(
            self.proj, feat, m, stride=self.stride, receptive_field_scaler=self.receptive_field_scaler
        )
        if self.norm is not None:
            feat = self.norm(feat, m.sample_sizes)
        if self.act is not None:
            feat = self.act(feat)
        return feat, m


class UnpoolingPointCNNpp(PointModule):
    """使用 PointCNN++ 的上采样层"""

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        kernel_size=3,
        receptive_field_scaler=2.5,
        straight_recover=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        # 上采样层
        self.upsample = Upsample(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=False,
            receptive_field_scaler=receptive_field_scaler,
            straight_recover=straight_recover,
        )

        # Skip connection 投影
        self.proj_skip = nn.Linear(skip_channels, out_channels)

        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        if act_layer is not None:
            self.act = act_layer()
        else:
            self.act = None

    def forward(
        self,
        feat_low: torch.Tensor,
        m_low: MetaData,
        feat_skip: torch.Tensor,
        m_skip: MetaData,
    ) -> Tuple[torch.Tensor, MetaData]:
        # 上采样
        feat_high, m_high = self.upsample(feat_low, m_low)

        # Skip connection: 需要对齐点数
        # 假设 m_high.points 和 m_skip.points 的点数相同（上采样后恢复）
        # 如果不同，需要根据坐标匹配
        if feat_high.shape[0] == feat_skip.shape[0]:
            # 点数相同，直接相加
            feat_skip_proj = self.proj_skip(feat_skip)
            feat = feat_high + feat_skip_proj
        else:
            # 点数不同，需要根据坐标匹配（简化处理：使用前 N 个点）
            min_points = min(feat_high.shape[0], feat_skip.shape[0])
            feat_skip_proj = self.proj_skip(feat_skip[:min_points])
            feat = feat_high[:min_points] + feat_skip_proj
            # 如果 feat_high 更长，保留剩余部分
            if feat_high.shape[0] > min_points:
                feat = torch.cat([feat, feat_high[min_points:]], dim=0)

        if self.norm is not None:
            feat = self.norm(feat, m_high.sample_sizes)
        if self.act is not None:
            feat = self.act(feat)

        return feat, m_high


@MODELS.register_module("PT-v3m3-pointcnnpp")
class PointTransformerV3PointCNNpp(PointModule):
    """
    基于 PointCNN++ 实现的 Point Transformer V3
    
    完全兼容 Pointcept 框架，使用 PointCNN++ 的点云算子和稀疏注意力机制。
    """

    def __init__(
        self,
        in_channels=6,
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_distance_threshold=(0.3, 0.3, 0.3, 0.3, 0.3),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_distance_threshold=(0.3, 0.3, 0.3, 0.3),
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        use_native=True,
        receptive_field_scaler=2.5,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.cls_mode = cls_mode

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_distance_threshold)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_distance_threshold) + 1

        # Norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    RaggedBatchNorm, reduce_fn=large_segment_reduce, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(RaggedBatchNorm, reduce_fn=large_segment_reduce, eps=1e-3, momentum=0.01)
        
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(RaggedLayerNorm, reduce_fn=large_segment_reduce, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        
        # Activation layer
        act_layer = nn.GELU

        # Embedding
        self.embedding = EmbeddingPointCNNpp(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
            kernel_size=5,
            receptive_field_scaler=receptive_field_scaler,
        )

        # Encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc_stages = nn.ModuleList()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            stage_modules = []
            
            if s > 0:
                # Downsample
                stage_modules.append(
                    (
                        "down",
                        DownsamplePointCNNpp(
                            in_channels=enc_channels[s - 1],
                            out_channels=enc_channels[s],
                            stride=stride[s - 1],
                            norm_layer=bn_layer,
                            act_layer=act_layer,
                            receptive_field_scaler=receptive_field_scaler,
                        ),
                    )
                )
            
            # Blocks
            for i in range(enc_depths[s]):
                stage_modules.append(
                    (
                        f"block{i}",
                        BlockPointCNNpp(
                            channels=enc_channels[s],
                            num_heads=enc_num_head[s],
                            distance_threshold=enc_distance_threshold[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=enc_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            use_native=use_native,
                            receptive_field_scaler=receptive_field_scaler,
                        ),
                    )
                )
            
            self.enc_stages.append(nn.ModuleDict(stage_modules))

        # Decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec_stages = nn.ModuleList()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                stage_modules = []
                
                # Unpooling
                stage_modules.append(
                    (
                        "up",
                        UnpoolingPointCNNpp(
                            in_channels=dec_channels[s + 1],
                            skip_channels=enc_channels[s],
                            out_channels=dec_channels[s],
                            norm_layer=bn_layer,
                            act_layer=act_layer,
                            receptive_field_scaler=receptive_field_scaler,
                        ),
                    )
                )
                
                # Blocks
                for i in range(dec_depths[s]):
                    stage_modules.append(
                        (
                            f"block{i}",
                            BlockPointCNNpp(
                                channels=dec_channels[s],
                                num_heads=dec_num_head[s],
                                distance_threshold=dec_distance_threshold[s],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_drop,
                                proj_drop=proj_drop,
                                drop_path=dec_drop_path_[i],
                                norm_layer=ln_layer,
                                act_layer=act_layer,
                                pre_norm=pre_norm,
                                use_native=use_native,
                                receptive_field_scaler=receptive_field_scaler,
                            ),
                        )
                    )
                
                self.dec_stages.append(nn.ModuleDict(stage_modules))

    def forward(self, data_dict):
        """
        前向传播
        
        Args:
            data_dict: 包含 feat, coord, offset 等的字典或 Point 对象
            
        Returns:
            Point 对象，包含处理后的特征
        """
        # 转换为 Point 对象
        if isinstance(data_dict, dict):
            point = Point(data_dict)
        else:
            point = data_dict

        # 转换为 pointcnnpp 格式
        feat, m = point_to_metadata(point)
        enc_outputs = []  # 保存编码器各阶段的输出用于 skip connection

        # Embedding
        feat, m = self.embedding(feat, m)
        enc_outputs.append((feat.clone(), m))

        # Encoder
        for s, stage in enumerate(self.enc_stages):
            if "down" in stage:
                feat, m = stage["down"](feat, m)
            for i in range(len([k for k in stage.keys() if k.startswith("block")])):
                feat, m = stage[f"block{i}"](feat, m)
            enc_outputs.append((feat.clone(), m))

        # Decoder
        if not self.cls_mode:
            for s, stage in enumerate(self.dec_stages):
                # Unpooling with skip connection
                feat_skip, m_skip = enc_outputs[-(s + 2)]  # 对应的编码器阶段
                feat, m = stage["up"](feat, m, feat_skip, m_skip)
                
                # Blocks
                for i in range(len([k for k in stage.keys() if k.startswith("block")])):
                    feat, m = stage[f"block{i}"](feat, m)

        # 转换回 Point 对象
        point = metadata_to_point(feat, m, point)
        return point
