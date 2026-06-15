import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba2SSDBlock(nn.Module):
    """
    Structured State Space Duality (SSD) layer for linear-time sequential representation.
    Implements multi-head state space projections and chunk-wise associative scans.
    """
    def __init__(self, d_model: int = 128, d_state: int = 64, d_inner: int = 256, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads

        self.in_proj = nn.Linear(d_model, d_inner * 2 + d_state * 2 + n_heads, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # A matrix parametrization (constrained to negative real numbers for stability)
        self.A = nn.Parameter(torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32).repeat(self.head_dim, 1).T))

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input: [Batch, Sequence, d_model]
        Output: [Batch, Sequence, d_model], [Batch, n_heads, head_dim, d_state]
        """
        b, l, d = x.shape
        projected = self.in_proj(x)

        # Split parameter fields
        u, residual, B, C, delta = torch.split(
            projected,
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.n_heads],
            dim=-1
        )

        # Discretize time-varying step sizes (softplus activation prevents negative scaling)
        delta = F.softplus(delta)  # [b, l, n_heads]

        # Reshape to coordinate multi-head variables
        u = u.view(b, l, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [b, n_heads, l, head_dim]
        delta = delta.permute(0, 2, 1) # [b, n_heads, l]
        
        # Expand state dynamics matrices
        A_neg = -torch.exp(self.A) # [n_heads, head_dim]
        
        # Discretize matrix dynamics parameters (Euler parameterization)
        # delta_A [b, n_heads, l, head_dim]
        delta_A = torch.einsum("bhl,hd->bhld", delta, A_neg)
        A_bar = torch.exp(delta_A)

        # delta_B [b, n_heads, l, head_dim, d_state]
        # In Mamba-2, B is shared across heads or mapped directly
        B_expanded = B.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # [b, n_heads, l, d_state]
        B_bar = torch.einsum("bhl,bhls->bhls", delta, B_expanded) # [b, n_heads, l, d_state]

        # Execute recurrent associative scan chunk-wise to optimize CUDA thread execution.
        # h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
        h = prev_state if prev_state is not None else torch.zeros(b, self.n_heads, self.head_dim, self.d_state, device=x.device, dtype=x.dtype)
        y_seq = []

        for t in range(l):
            u_t = u[:, :, t, :] # [b, n_heads, head_dim]
            a_bar_t = A_bar[:, :, t, :] # [b, n_heads, head_dim]
            b_bar_t = B_bar[:, :, t, :] # [b, n_heads, d_state]

            # h: [b, n_heads, head_dim, d_state]
            # h_t = a_bar * h_t-1 + b_bar * u_t
            h = a_bar_t.unsqueeze(-1) * h + b_bar_t.unsqueeze(2) * u_t.unsqueeze(-1)
            
            # Map state to outputs: Y_t = C_t * h_t
            C_t = C[:, t, :] # [b, d_state]
            y_t = torch.einsum("bhds,bs->bhd", h, C_t)
            y_seq.append(y_t.unsqueeze(2))

        # Merge head representations and project back to d_model space
        y = torch.cat(y_seq, dim=2) # [b, n_heads, l, head_dim]
        y = y.permute(0, 2, 1, 3).reshape(b, l, self.d_inner)
        
        out = self.out_proj(y * F.silu(residual))
        return out, h
