import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * dh)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(q_in))
        K = self._split_heads(self.W_k(k_in))
        V = self._split_heads(self.W_v(v_in))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = self.drop(F.softmax(scores, dim=-1))
        out = attn @ V
        out = self.W_o(self._merge_heads(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        lb_coef: float = 1e-2,
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.lb_coef = lb_coef

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff, dropout) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, model_dim = x.shape

        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_logits, topk_expert_indices = torch.topk(
            gate_logits, k=self.top_k, dim=-1
        )
        topk_weights = F.softmax(topk_logits, dim=-1)

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=2,
        )

        selected_expert_outputs = torch.gather(
            expert_outputs,
            dim=2,
            index=topk_expert_indices
            .unsqueeze(-1)
            .expand(batch_size, seq_len, self.top_k, model_dim),
        )

        output = (selected_expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)

        expert_one_hot = F.one_hot(
            topk_expert_indices, num_classes=self.n_experts
        ).to(gate_probs.dtype)

        expert_load = expert_one_hot.sum(dim=2).mean(dim=(0, 1))
        expert_importance = gate_probs.mean(dim=(0, 1))

        aux_loss = self.lb_coef * (
            self.n_experts * torch.sum(expert_load * expert_importance)
        )

        return output, aux_loss



class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        n_experts: int = 4,
        top_k: int = 2,
        lb_coef: float = 1e-2,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = MoEFeedForward(d_model, d_ff, n_experts=n_experts, top_k=top_k, dropout=dropout, lb_coef=lb_coef)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        mem_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, h, attn_mask=self_attn_mask, key_padding_mask=self_key_padding_mask))

        h = self.ln2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, attn_mask=None, key_padding_mask=mem_key_padding_mask))

        h = self.ln3(x)
        ffn_out, aux = self.ffn(h)
        x = x + self.drop(ffn_out)

        return x, aux
