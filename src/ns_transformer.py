import numpy as np
import torch
import torch.nn as nn

from src.embed import DataEmbedding
from src.layers import Encoder, EncoderLayer, DecoderLayer, Decoder, AttentionLayer
from src.utils import TriangularCausalMask
from math import sqrt

#########################################
# DSAttiontion
#########################################
class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


#########################################
# Model
#########################################
class Model(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # 비정상성 보정용 projector
        self.tau_learner   = Projector(
            enc_in=configs.enc_in,
            seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims,
            hidden_layers=configs.p_hidden_layers,
            output_dim=1
        )
        self.delta_learner = Projector(
            enc_in=configs.enc_in,
            seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims,
            hidden_layers=configs.p_hidden_layers,
            output_dim=configs.seq_len
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # x_raw: [B, L, E]
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # decoder에서 쓸 부분
        x_dec_new = torch.cat(
            [x_enc[:, -self.label_len:, :],
             torch.zeros_like(x_dec[:, -self.pred_len:, :])],
            dim=1
        ).to(x_enc.device).clone()

        # 비정상성 보정
        tau = self.tau_learner(x_raw, std_enc).exp()  # [B, 1]
        delta = self.delta_learner(x_raw, mean_enc)   # [B, seq_len]

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        # Decoder
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask,
                               cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # 역정규화
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


#########################################
# Projector
#########################################
class Projector(nn.Module):
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size,
                                     padding=padding, padding_mode='circular', bias=False)
        self.hidden_dims = hidden_dims
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.backbone = None  # 초기엔 None으로 설정

    def build_mlp(self, in_features):
        # self.hidden_dims를 복사 후, self.hidden_layers에 맞게 길이를 조정합니다.
        hidden_dims = list(self.hidden_dims)
        if self.hidden_layers > len(hidden_dims):
            # 부족한 층은 마지막 값으로 채움
            hidden_dims.extend([hidden_dims[-1]] * (self.hidden_layers - len(hidden_dims)))
        elif self.hidden_layers < len(hidden_dims):
            # 필요한 만큼만 사용
            hidden_dims = hidden_dims[:self.hidden_layers]

        layers = [nn.Linear(in_features, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x, stats):
        batch_size, seq_len, E = x.shape

        x = self.series_conv(x)           # [B, 1, E]
        x = torch.cat([x, stats], dim=1)  # [B, 2, E]
        x = x.view(batch_size, -1)        # [B, 2*E]

        # 첫 forward 호출 시 MLP 동적으로 생성
        if self.backbone is None:
            in_features = x.shape[-1]  # = 2 * E
            self.backbone = self.build_mlp(in_features)

        return self.backbone(x)

