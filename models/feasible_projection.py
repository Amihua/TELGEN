"""
Feasible Projection Layer for ensuring LP solution feasibility
"""
import torch
import torch.nn as nn


class FeasibleProjectionLayer(nn.Module):
    """
    可微分的可行域投影层
    
    将预测值投影到可行域（默认） {x: Ax <= b, x >= 0}

    - 传统模式（legacy）：仅支持 Ax<=b（上界型不等式）+ 非负，用简单缩放：
        x' = alpha * x

    - 扩展模式（用于 TE-IPM SD equality）：
        当检测到 SD 约束被编码为 sum(x_block)=1（即存在 1/-1 的两类块约束）时，
        采用“交替投影”：
          1) 对每个 SD block 做 simplex 投影（sum=1, x>=0）
          2) 对 capacity 约束做缩放（alpha<=1 使 Ax<=b）
        重复若干次（projection_iterations）。
      这样能同时处理 “等式（通过 simplex）” 和 “上界不等式（通过缩放）”。
    """
    def __init__(self, projection_iterations=10, step_size=0.1):
        super().__init__()
        self.projection_iterations = projection_iterations
        self.step_size = step_size

    @staticmethod
    def _project_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
        """
        Project v onto simplex {x>=0, sum x = z}.
        v: [B, k]
        """
        # Wang & Carreira-Perpinan (2013)
        B, k = v.shape
        u, _ = torch.sort(v, dim=1, descending=True)
        cssv = torch.cumsum(u, dim=1) - z
        ind = torch.arange(1, k + 1, device=v.device, dtype=v.dtype).view(1, -1)
        cond = u - cssv / ind > 0
        # rho: last True index per row
        rho = cond.sum(dim=1).clamp(min=1)
        rho_idx = (rho - 1).to(torch.long)
        theta = cssv.gather(1, rho_idx.view(-1, 1)) / rho.to(v.dtype).view(-1, 1)
        w = torch.clamp(v - theta, min=0.0)
        return w

    @staticmethod
    def _infer_k_from_A(A: torch.Tensor) -> int | None:
        """
        Infer k (paths per SD) by finding the most frequent nonzero-count among rows whose nonzeros are all +/-1.
        Returns None if cannot infer.
        """
        with torch.no_grad():
            mask = A != 0
            counts = mask.sum(dim=1)  # [m]
            # rows where all nonzeros are +/-1
            absA = A.abs()
            # For each row: max |A_ij| over nonzeros and min |A_ij| over nonzeros
            # Implement via masked fill.
            absA_masked = absA.masked_fill(~mask, 0.0)
            row_max = absA_masked.max(dim=1).values
            absA_min_masked = absA.masked_fill(~mask, 1e9)
            row_min = absA_min_masked.min(dim=1).values
            sd_like = (counts > 0) & (row_max <= 1.0 + 1e-6) & (row_min >= 1.0 - 1e-6)

            cand = counts[sd_like]
            if cand.numel() == 0:
                return None
            # k is the mode of cand (integer counts)
            cand_i = cand.to(torch.long)
            max_c = int(cand_i.max().item())
            hist = torch.bincount(cand_i, minlength=max_c + 1)
            # ignore 0/1 counts
            if hist.numel() <= 2:
                return None
            hist[:2] = 0
            k = int(hist.argmax().item())
            return k if k > 1 else None
    
    def forward(self, x_pred, A, b):
        """
        可行域投影：
        - 若检测到 SD equality（block simplex）结构，则交替投影；
        - 否则使用 legacy 简单缩放投影（仅保证 Ax<=b + x>=0）。
        
        Args:
            x_pred: [batch_size, num_vars] 或 [num_vars]
            A: [num_constraints, num_vars]
            b: [num_constraints] 或 [batch_size, num_constraints]
        """
        x = x_pred.clone()

        # 处理 batch 维度
        is_batch = x.dim() > 1
        if not is_batch:
            x = x.unsqueeze(0)
            if b.dim() == 1:
                b = b.unsqueeze(0)

        batch_size = x.shape[0]

        # 保持非负
        x = torch.clamp(x, min=0.0)

        if A.dim() != 2:
            raise NotImplementedError("Sparse A matrix not supported in projection")

        # Expand b
        if b.dim() == 1:
            b_expanded = b.unsqueeze(0).expand(batch_size, -1)
        else:
            b_expanded = b

        # Try to infer SD block size k and apply alternating projection if it looks like SD equality is present.
        k = self._infer_k_from_A(A)
        use_alt = False
        if k is not None and k > 1 and (A.shape[1] % k == 0):
            # If we also see negative b entries (from -sum <= -1), it's a strong signal.
            if (b_expanded < 0).any().item():
                use_alt = True

        if use_alt:
            num_sd = A.shape[1] // k

            # Identify SD-like rows to exclude them from capacity scaling.
            mask = A != 0
            counts = mask.sum(dim=1)
            absA = A.abs()
            absA_masked = absA.masked_fill(~mask, 0.0)
            row_max = absA_masked.max(dim=1).values
            absA_min_masked = absA.masked_fill(~mask, 1e9)
            row_min = absA_min_masked.min(dim=1).values
            sd_like_rows = (counts == k) & (row_max <= 1.0 + 1e-6) & (row_min >= 1.0 - 1e-6)
            cap_rows = ~sd_like_rows

            # alternating projection
            iters = max(int(self.projection_iterations), 1)
            for _ in range(iters):
                # 1) simplex per SD block: enforce sum==1, x>=0
                xb = x.view(batch_size, num_sd, k)
                xb = self._project_simplex(xb.reshape(batch_size * num_sd, k), z=1.0).view(batch_size, num_sd, k)
                x = xb.reshape(batch_size, num_sd * k)

                # 2) capacity scaling: enforce only upper-type constraints (cap_rows) by alpha<=1
                if cap_rows.any():
                    A_cap = A[cap_rows]  # [m_cap, n]
                    b_cap = b_expanded[:, cap_rows]  # [B, m_cap]
                    Ax_cap = torch.matmul(A_cap, x.T).T  # [B, m_cap]
                    viol = Ax_cap > b_cap
                    if viol.any():
                        eps = 1e-12
                        ratios = b_cap / (Ax_cap + eps)
                        ratios = torch.where(viol, ratios, torch.ones_like(ratios))
                        # scaling down only
                        ratios = torch.clamp(ratios, min=0.0, max=1.0)
                        alpha = torch.min(ratios, dim=1, keepdim=True).values
                        alpha = torch.clamp(alpha, max=1.0)
                        x = x * alpha
                x = torch.clamp(x, min=0.0)
        else:
            # legacy: simple scaling for Ax<=b
            Ax = torch.matmul(A, x.T).T  # [batch_size, num_constraints]
            violations = Ax > b_expanded
            if violations.any():
                eps = 1e-12
                ratios = b_expanded / (Ax + eps)
                ratios = torch.where(violations, ratios, torch.ones_like(ratios))
                ratios = torch.clamp(ratios, min=0.0, max=1.0)
                alpha = torch.min(ratios, dim=1, keepdim=True).values
                alpha = torch.clamp(alpha, max=1.0)
                x = x * alpha

        if not is_batch:
            x = x.squeeze(0)

        return x


def extract_A_matrix_from_data(data):
    """
    从HeteroData中重建A矩阵
    
    Args:
        data: HeteroData对象，包含A_row, A_col, A_val等属性
        
    Returns:
        A: [num_constraints, num_vars] 密集矩阵
    """
    # 从稀疏表示重建密集矩阵
    row = data.A_row
    col = data.A_col
    val = data.A_val
    
    # 创建稀疏张量
    indices = torch.stack([row, col], dim=0)
    A_sparse = torch.sparse_coo_tensor(
        indices,
        val,
        size=(data.A_num_row, data.A_num_col)
    )
    
    # 转换为密集矩阵
    A = A_sparse.to_dense()
    
    return A


def extract_A_matrix_batch(data_batch):
    """
    从批次数据中提取A矩阵（处理batch情况）
    
    Args:
        data_batch: Batch对象，包含多个图
        
    Returns:
        A_list: 每个图的A矩阵列表
    """
    A_list = []
    
    # 将batch分解为单个图
    from torch_geometric.data import Batch
    data_list = Batch.to_data_list(data_batch)
    
    for data in data_list:
        A = extract_A_matrix_from_data(data)
        A_list.append(A)
    
    return A_list

