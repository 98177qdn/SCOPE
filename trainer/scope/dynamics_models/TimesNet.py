from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)   # [T_freq]
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff: int,
        top_k: int = 2,
        num_kernels: int = 6,
    ):
        """
        原本 TimesBlock(configs) 改成显式参数版本。
        seq_len, pred_len 用来确定 padding 后的总长度 (seq_len + pred_len)。
        """
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        """
        x: [B, T, N] （T 一般等于 seq_len + pred_len，N=d_model）
        """
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)  # period_list: [k]

        res = []
        total_len = self.seq_len + self.pred_len

        for i in range(self.k):
            period = period_list[i]
            # padding
            if total_len % period != 0:
                length = (total_len // period + 1) * period
                padding_len = length - total_len
                padding = torch.zeros(
                    [x.shape[0], padding_len, x.shape[2]], device=x.device, dtype=x.dtype
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)   # [B, T_total, N, k]

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)  # [B, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, total_len, N, 1)
        res = torch.sum(res * period_weight, dim=-1)     # [B, T_total, N]

        # residual connection（只在前 seq_len 上有原始 x，后面 pred_len 部分 residual 实际也来自 x 的扩展）
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq

    DLinear 风格接口：
      __init__(seq_len, pred_len, individual, enc_in, ...)
      forward(x): x [B, seq_len, enc_in] -> [B, pred_len, c_out]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        individual: bool,
        enc_in: int,
        # TimesNet / embedding 超参数
        d_model: int = 64,
        d_ff: int = 256,
        e_layers: int = 2,
        top_k: int = 2,
        num_kernels: int = 6,
        c_out: Optional[int] = None,
        embed: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
        # 其他任务相关（如果只做预测可以忽略）
        task_name: str = "long_term_forecast",
        num_class: Optional[int] = None,
    ):
        super(TimesNet, self).__init__()

        # --- 与 DLinear 对齐的几个基础属性 ---
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in

        self.task_name = task_name
        self.layer = e_layers
        self.c_out = c_out if c_out is not None else enc_in  # 默认输出维度 = 输入维度

        # embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        # TimesBlocks
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, d_model, d_ff, top_k=top_k, num_kernels=num_kernels)
            for _ in range(e_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

        # 预测 / 其他任务的头部
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            # 把时间维从 seq_len 投影到 seq_len + pred_len
            self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
            self.projection = nn.Linear(d_model, self.c_out, bias=True)
        elif self.task_name in ["imputation", "anomaly_detection"]:
            self.projection = nn.Linear(d_model, self.c_out, bias=True)
        elif self.task_name == "classification":
            assert num_class is not None, "classification 任务需要指定 num_class"
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * self.seq_len, num_class)
        else:
            # 默认当作预测任务
            self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
            self.projection = nn.Linear(d_model, self.c_out, bias=True)

    # ========= 各种任务的子函数（保留原逻辑） =========
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: [B, L, N] （L = seq_len, N = enc_in）
        返回: [B, L + pred_len, c_out]
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()   # [B, 1, N]
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]

        # 先扩展时间维到 L + pred_len
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L+P, d_model]

        # TimesNet blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))  # [B, L+P, d_model]

        # project back to feature dimension
        dec_out = self.projection(enc_out)  # [B, L+P, c_out]

        # De-Normalization from Non-stationary Transformer
        total_len = self.seq_len + self.pred_len
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        )
        return dec_out  # [B, L+P, c_out]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        total_len = self.seq_len + self.pred_len
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        )
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        total_len = self.seq_len + self.pred_len
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        )
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        output = self.act(enc_out)
        output = self.dropout_layer(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    # ========= DLinear 风格 forward =========
    def forward(self, x):
        """
        DLinear 接口：
          x: [B, seq_len, enc_in]
        默认执行预测任务（long_term_forecast 风格）：
          返回: [B, pred_len, c_out]
        """
        dec_out = self.forecast(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
        # 只取最后 pred_len 步
        return dec_out[:, -self.pred_len:, :]
