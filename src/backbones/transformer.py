import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, '`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(~key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(~attention_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        # attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        # self.dropout = build_dropout_layer(dropout)
        self.norm_out = nn.LayerNorm(d_model)

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(d_model * 2, d_model), nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(d_model)
        )

    def forward(
            self,
            input_states,
            memory_weights=None,
            memory_masks=None,
            attention_factors=None,
            attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            input_states,
            input_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        norm_states = self.norm_out(hidden_states + input_states)
        output_states = self.output(norm_states)
        return output_states, attention_scores


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(CrossAttention, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        # self.dropout = build_dropout_layer(dropout)
        self.norm_out = nn.LayerNorm(d_model)

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(d_model * 2, d_model), nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(d_model)
        )

    def forward(
            self,
            input_states,
            memory_states,
            memory_weights=None,
            memory_masks=None,
            attention_factors=None,
            attention_masks=None,
    ):
        input_states, self_attention_scores = self.self_attention(
            input_states,
            input_states,
            input_states,
            key_weights=None,
            key_masks=None,
            attention_factors=None,
            attention_masks=None,
        )
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        norm_states = self.norm_out(hidden_states + input_states)
        output_states = self.output(norm_states)
        return output_states, attention_scores


class PointTransformer(nn.Module):
    def __init__(self, d_model, c_model, num_heads, coarse):
        super(PointTransformer, self).__init__()
        self.coarse = coarse
        self.d_model = float(d_model)
        self.self_attention = SelfAttention(d_model=d_model, c_model=c_model, num_heads=num_heads, coarse=self.coarse)
        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads)
        self.f_linear = nn.Linear(d_model, d_model)
        self.c_linear = nn.Linear(d_model, d_model)

    def forward(self, srcf_fts, reff_fts, srcc_fts=None, refc_fts=None, src_masks=None, ref_masks=None):
        # print(srcf_fts.shape)
        src_fts, src_fscore = self.self_attention(input_states=srcf_fts, course_states=srcc_fts, memory_masks=src_masks)
        ref_fts, ref_fscore = self.self_attention(input_states=reff_fts, course_states=refc_fts, memory_masks=ref_masks)

        src_cfts, src_cscore = self.cross_attention(src_fts, ref_fts, memory_masks=ref_masks)  # (b,n,c)
        ref_cfts, ref_cscore = self.cross_attention(ref_fts, src_fts, memory_masks=src_masks)  # (b,m,c)

        src_cfts = self.f_linear(srcf_fts) + self.c_linear(src_cfts)
        ref_cfts = self.f_linear(reff_fts) + self.c_linear(ref_cfts)
        scores = torch.einsum('bnd,bmd->bnm', ref_cfts, src_cfts) / self.d_model
        return scores


class CoverageTransformer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CoverageTransformer, self).__init__()
        self.d_model = float(d_model)
        self.self_attention = SelfAttention(d_model=d_model, num_heads=num_heads)
        self.f_linear = nn.Linear(d_model, d_model)

    def forward(self, fts, masks=None):
        attention_output, scores = self.self_attention(fts, memory_masks=masks)  # (b,n,c)
        output = self.f_linear(attention_output)
        return output, scores.mean(dim=1)


if __name__ == "__main__":
    data = torch.randn((2, 5, 512))
    transformer = CoverageTransformer(d_model=512, num_heads=4)
    output, scores = transformer(data)
    print("test")
