from torch import nn
import torch


# Здесь я какое-то время общался с LLM, чтобы разобраться dtype, так как хотел чтобы LN считался в fp32, а остальная часть пайплайна не обязательно
class VisionAdapter(nn.Module):
    """
    Input:  vision last_hidden_state: (B, S, Dv), token 0 is CLS
    Output: visual tokens in Qwen space: (B, T, Dq)
    """

    def __init__(self, vision_dim, qwen_dim, n_tokens=16, ln_eps=1e-5):
        super().__init__()
        self.n_tokens = n_tokens

        self.conv = nn.Conv1d(vision_dim, vision_dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)

        self.ln1 = nn.LayerNorm(vision_dim, eps=ln_eps)
        self.fc  = nn.Linear(vision_dim, qwen_dim)
        self.ln2 = nn.LayerNorm(qwen_dim, eps=ln_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype  # keep pipeline dtype (fp16/bf16) at the boundary

        # drop CLS
        y = x[:, 1:, :].transpose(1, 2).contiguous()

        y = self.conv(y)
        y = self.pool(y)                 # fixed token count
        y = y.transpose(1, 2).contiguous()  # (B, T, Dv)

        y = self.ln1(y)                  # autocast will pick dtype per-op
        y = self.fc(y)
        y = self.ln2(y)

        return y.to(out_dtype)           # single, explicit boundary cast
