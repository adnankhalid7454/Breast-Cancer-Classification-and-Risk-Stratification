import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet50, ResNet50_Weights


class PlaneAggregation2D(nn.Module):
    """
    Convert a 3D volume tensor x of shape (B, C, D, H, W) into three 2D plane images:
    1. axial    = aggregate over D  -> (B, C, H, W)
    2. coronal  = aggregate over H  -> (B, C, D, W)
    3. sagittal = aggregate over W  -> (B, C, D, H)

    Each plane image is resized to image_size x image_size, then passed to a 2D CNN.
    The CNN output is projected into transformer tokens, one token per plane.
    """
    def __init__(
        self,
        in_channels: int = 3,
        emb_size: int = 256,
        image_size: int = 128,
        agg_mode: str = "max",
    ):
        super().__init__()
        self.image_size = image_size
        self.agg_mode = agg_mode.lower()

        if self.agg_mode not in {"mean", "max"}:
            raise ValueError("agg_mode must be either 'mean' or 'max'")

        self.cnn_2d = resnet50(weights=ResNet50_Weights.DEFAULT)
        if in_channels != 3:
            self.cnn_2d.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.cnn_2d.fc = nn.Identity()

        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_size),
        )

    def _aggregate(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if self.agg_mode == "mean":
            return x.mean(dim=dim)
        return x.max(dim=dim).values

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        axial = self._aggregate(x, dim=2)     # (B, C, H, W)
        coronal = self._aggregate(x, dim=3)   # (B, C, D, W)
        sagittal = self._aggregate(x, dim=4)  # (B, C, D, H)

        axial = self._resize(axial)
        coronal = self._resize(coronal)
        sagittal = self._resize(sagittal)

        # fixed token order: [coronal, sagittal, axial]
        planes = torch.stack([coronal, sagittal, axial], dim=1)  # (B, 3, C, S, S)
        b, n, c, h, w = planes.shape
        planes = planes.view(b * n, c, h, w)

        feats = self.cnn_2d(planes)           # (B*3, 2048)
        tokens = self.proj(feats)             # (B*3, emb_size)
        tokens = tokens.view(b, n, -1)        # (B, 3, emb_size)
        return tokens


class EmbeddingLayer2D(nn.Module):
    """
    Build transformer input with one CLS token and three plane tokens.
    Sequence order: [CLS, coronal, sagittal, axial]
    """
    def __init__(self, emb_size: int = 256):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.coronal_plane = nn.Parameter(torch.randn(1, 1, emb_size))
        self.sagittal_plane = nn.Parameter(torch.randn(1, 1, emb_size))
        self.axial_plane = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(1, 4, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, emb_size)
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)

        x = x.clone()
        x[:, 0:1, :] = x[:, 0:1, :] + self.coronal_plane
        x[:, 1:2, :] = x[:, 1:2, :] + self.sagittal_plane
        x[:, 2:3, :] = x[:, 2:3, :] + self.axial_plane

        x = torch.cat([cls, x], dim=1)        # (B, 4, emb_size)
        x = x + self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(
            self.qkv(x),
            "b n (three h d) -> three b h n d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        energy = energy / math.sqrt(self.head_dim)

        if mask is not None:
            fill_value = torch.finfo(energy.dtype).min
            energy = energy.masked_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum("bhqk,bhkd->bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return x + res


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 2, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 256,
        drop_p: float = 0.0,
        forward_expansion: int = 2,
        forward_drop_p: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=forward_drop_p,
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 8, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 256, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, 0]
        return self.linear(cls_token)


class M3T(nn.Sequential):
    """
    2D aggregated version of M3T.

    Input:  (B, C, D, H, W)
    Step 1: Aggregate slices into 3 plane images
    Step 2: Extract CNN features from the 3 images
    Step 3: Convert CNN features to 3 transformer tokens
    Step 4: Transformer encoder over [CLS, coronal, sagittal, axial]
    Step 5: Classification from CLS token
    """
    def __init__(
        self,
        in_channels: int = 3,
        emb_size: int = 256,
        depth: int = 8,
        n_classes: int = 2,
        image_size: int = 128,
        agg_mode: str = "mean",
        **kwargs,
    ):
        super().__init__(
            PlaneAggregation2D(
                in_channels=in_channels,
                emb_size=emb_size,
                image_size=image_size,
                agg_mode=agg_mode,
            ),
            EmbeddingLayer2D(emb_size=emb_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size=emb_size, n_classes=n_classes),
        )


if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128, 128)
    model = M3T(in_channels=3, n_classes=2, image_size=128, agg_mode="mean")
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
