import torch
import torch.nn as nn
class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class MaskedMultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)

    def forward(self, x, key_padding_mask=None):
        x = self.layer_norm(x)
        batch_size, seq_len, _ = x.size()
        casual_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=casual_mask,  # Causal mask for self-attention
            key_padding_mask=key_padding_mask,  # Padding mask
            need_weights=True,
            average_attn_weights=False,
        )

        return attn_output


class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()

        self.masked_msa_block = MaskedMultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, key_padding_mask=None):
        # print(f"Before self-attention: {x.isnan().any()}")

        attn_output = self.masked_msa_block(x, key_padding_mask)
        x_residual1 = attn_output + x
        # print(f"After self-attention: {x_residual1.isnan().any()}")

        # Apply Feed-Forward block (MLP) with residual connection
        mlp_output = self.mlp_block(x_residual1)
        x_residual2 = mlp_output + x_residual1
        # print(f"After feed-forward: {x_residual2.isnan().any()}")

        return x_residual2

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size=114, seq_len=3, embedding_dim=128, num_heads=4, mlp_size=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.token_in = nn.Linear(vocab_size, embedding_dim, bias=False)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x_onehot):
        B, T, V = x_onehot.shape
        assert V == self.vocab_size
        assert T == self.seq_len

        x = self.token_in(x_onehot) + self.pos_emb[:, :T, :]

        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + attn_out

        h = self.ln2(x)
        x = x + self.mlp(h)

        logits = self.head(x)
        return logits
