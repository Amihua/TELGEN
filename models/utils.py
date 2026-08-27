from typing import Optional, List

import torch
from torch.nn import Sequential, BatchNorm1d, LayerNorm, InstanceNorm1d, ReLU, Dropout
from torch_geometric.nn import Linear

# 尝试从 torch_scatter 导入 broadcast；如果失败，则使用简化版 broadcast
try:  # pragma: no cover - 环境兼容逻辑
    from torch_scatter.utils import broadcast  # type: ignore
except Exception:
    def broadcast(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
        """
        简化版 broadcast：
        - 如果 index 已经和 src 形状相同，直接返回；
        - 如果 index 是一维，沿给定 dim 展开到 src 的形状。
        仅用于本文件中的 scatter_sum。
        """
        if index.dim() == src.dim():
            return index
        if index.dim() != 1:
            raise ValueError(f"Unsupported index shape for broadcast fallback: {index.shape}")
        shape = [1] * src.dim()
        shape[dim] = -1
        index_expanded = index.view(*shape)
        index_expanded = index_expanded.expand_as(src)
        return index_expanded


# originally this was in-place, for torch.vmap(torch.func.jacrec) reasons we need non in-place version
# https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/scatter.html#scatter
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return torch.scatter_add(out, dim, index, src)
    else:
        return torch.scatter_add(out, dim, index, src)


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super().__init__(*m)
