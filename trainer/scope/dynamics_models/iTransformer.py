import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625

    接口对齐 DLinear：
      __init__(seq_len, pred_len, individual, enc_in, ...)
      forward(x): x [B, seq_len, enc_in] -> [B, pred_len, enc_in]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        individual: bool,
        enc_in: int,
        # 下面这些是原 configs 里的超参数，给出默认值，按需改：
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        factor=5,
        dropout=0.1,
        activation: str = "gelu",
        embed: str = "timeF",
        freq: str = "h",
        # 任务类型，默认为长序列预测
        task_name: str = "long_term_forecast",
        # 分类任务才会用到：
        num_class: int = None,
    ):
        super(iTransformer, self).__init__()

        # ---- 与 DLinear 一致的基础属性 ----
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in

        # ---- 原 configs 内容 ----
        self.task_name = task_name
        self.d_model = d_model

        # Embedding（iTransformer 的倒置嵌入）
        # 注意原代码是 DataEmbedding_inverted(configs.seq_len, ...)，这里就用 seq_len 保持一致
        self.enc_embedding = DataEmbedding_inverted(
            seq_len, d_model, embed, freq, dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Decoder / projection
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            # d_model -> pred_len
            self.projection = nn.Linear(d_model, pred_len, bias=True)

        if self.task_name == "imputation":
            self.projection = nn.Linear(d_model, seq_len, bias=True)

        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(d_model, seq_len, bias=True)

        if self.task_name == "classification":
            assert num_class is not None, "classification 任务需要指定 num_class"
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(dropout)
            # enc_in 对应变量数
            self.projection = nn.Linear(d_model * enc_in, num_class)

    # ====== 各种任务的子函数（基本沿用原版） ======
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: [B, L, N]，与 DLinear 对齐: L=seq_len, N=enc_in
        返回: [B, pred_len, N]
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, N]
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [B, L, d_model]

        # d_model -> pred_len, 然后换轴
        dec_out = self.projection(enc_out).permute(0, 2, 1)  # [B, pred_len, L]
        dec_out = dec_out[:, :, :N]  # 剪成 [B, pred_len, N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out  # [B, pred_len, N]

    def imputation(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        _, L, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        _, L, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout_layer(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, enc_in * d_model)
        output = self.projection(output)              # (batch_size, num_classes)
        return output

    # ====== 对齐 DLinear 的 forward 接口 ======
    def forward(self, x):
        """
        DLinear 接口：
          x: [B, seq_len, enc_in]
        默认执行 long_term_forecast / short_term_forecast 路径，
        返回: [B, pred_len, enc_in]
        """
        # 如果你以后想根据 task_name 自动切换，这里也可以写分支；
        # 目前为了和 DLinear 对齐，只做预测任务。
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
            # 原 forward 是 dec_out[:, -self.pred_len:, :]，这里 dec_out 本来就是 [B, pred_len, C]
            return dec_out  # [B, pred_len, enc_in]

        elif self.task_name == "imputation":
            return self.imputation(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)

        elif self.task_name == "anomaly_detection":
            return self.anomaly_detection(x)

        elif self.task_name == "classification":
            return self.classification(x, x_mark_enc=None)

        # 默认当作预测
        return self.forecast(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
