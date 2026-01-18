import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from layers.Autoformer_EncDec import series_decomp


class SeriesfitLinear(nn.Module):
    """
    原本依赖 configs 的版本，改成显式参数：
      linear_in:  Cin
      w_len:      窗口长度
      linear_out: Cout
      series_fit_add_res: 是否做 seasonal + trend 残差结构
      moving_avg: series_decomp 的滑动平均核
      linear_dropout: 是否在非残差模式下使用 dropout
    输入 x 默认按 PatchTST 的输出：[B, T, Cin]
    输出 out: [B, N, Cout]，N = T - w_len + 1
    """
    def __init__(
        self,
        linear_in: int,
        w_len: int,
        linear_out: int,
        series_fit_add_res: bool = False,
        moving_avg: int = 25,
        linear_dropout: bool = False,
    ):
        super(SeriesfitLinear, self).__init__()
        self.input_channels = linear_in      # Cin
        self.w_len = w_len                   # 窗口长度
        self.output_dim = linear_out         # Cout

        self.series_fit_add_res = series_fit_add_res
        self.linear_dropout = linear_dropout

        if self.series_fit_add_res:
            self.Linear_Seasonal = nn.Linear(self.input_channels * self.w_len, self.output_dim)
            self.Linear_Trend = nn.Linear(self.input_channels * self.w_len, self.output_dim)
            self.decompsition = series_decomp(moving_avg)
            print("SeriesfitLinear: 使用 seasonal + trend 残差形式")
        else:
            self.linear = nn.Linear(self.input_channels * self.w_len, self.output_dim)

        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        输入:  x, shape = [B, T, Cin]  （和 DLinear / PatchTST 一致：时间在中间）
        输出:  out, shape = [B, N, Cout]，其中 N = T - w_len + 1（滑窗步长=1）
        """
        if self.series_fit_add_res:
            # x: [B, T, Cin]
            seasonal_init, trend_init = self.decompsition(x)  # [B, T, Cin]
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            # 现在: [B, Cin, T]

            # Seasonal 分支
            B, C, T = seasonal_init.shape
            x_unfold = seasonal_init.unfold(dimension=2, size=self.w_len, step=1)      # [B, C, N, w]
            x_unfold = x_unfold.permute(0, 2, 1, 3).reshape(B, -1, C * self.w_len)     # [B, N, C*w]
            seasonal_out = self.Linear_Seasonal(x_unfold)                               # [B, N, Cout]

            # Trend 分支
            B, C, T = trend_init.shape
            x_unfold = trend_init.unfold(dimension=2, size=self.w_len, step=1)          # [B, C, N, w]
            x_unfold = x_unfold.permute(0, 2, 1, 3).reshape(B, -1, C * self.w_len)     # [B, N, C*w]
            trend_out = self.Linear_Trend(x_unfold)                                     # [B, N, Cout]

            out = seasonal_out + trend_out                                             # [B, N, Cout]
        else:
            # 简单线性分支
            # x: [B, T, Cin] -> [B, Cin, T]
            x = x.permute(0, 2, 1)
            B, C, T = x.shape
            x_unfold = x.unfold(dimension=2, size=self.w_len, step=1)                  # [B, C, N, w]
            x_unfold = x_unfold.permute(0, 2, 1, 3).reshape(B, -1, C * self.w_len)    # [B, N, C*w]
            out = self.linear(x_unfold)                                                # [B, N, Cout]
            if self.linear_dropout:
                out = self.dropout(out)

        return out


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)     # [bs, nvars, nf]
        x = self.linear(x)      # [bs, nvars, target_window]
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf

    现在接口对齐 DLinear：
      __init__(seq_len, pred_len, individual, enc_in, ...)
      forward(x): x [B, seq_len, enc_in] -> [B, pred_len, enc_out]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        individual: bool,
        enc_in: int,
        # PatchTST / Transformer 超参数（原 configs）
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        factor: int = 5,
        dropout: float = 0.1,
        activation: str = "gelu",
        # patch 参数
        patch_len: int = 4,
        stride: int = 2,
        # 任务类型（默认 long_term_forecast）
        task_name: str = "long_term_forecast",
        num_class: int = None,
        # 是否使用 SeriesfitLinear 做 enc_in -> c_out 的转换
        trans_enc_in_to_c_out: bool = False,
        c_out: int = None,
        # SeriesfitLinear 的相关参数
        sf_w_len: int = 25,
        sf_moving_avg: int = 25,
        sf_series_fit_add_res: bool = False,
        sf_linear_dropout: bool = False,
    ):
        """
        seq_len:   输入序列长度
        pred_len:  预测长度
        enc_in:    通道数（变量数）
        """
        super().__init__()

        # DLinear 对齐属性
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in

        self.task_name = task_name
        self.d_model = d_model
        self.trans_enc_in_to_c_out = trans_enc_in_to_c_out
        self.c_out = c_out if c_out is not None else enc_in

        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, factor, attention_dropout=dropout, output_attention=False
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
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            ),
        )

        # Prediction Head
        # patch_num = int((seq_len - patch_len) / stride + 2)
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.head = FlattenHead(enc_in, self.head_nf, pred_len, head_dropout=dropout)
        elif self.task_name in ["imputation", "anomaly_detection"]:
            self.head = FlattenHead(enc_in, self.head_nf, seq_len, head_dropout=dropout)
        elif self.task_name == "classification":
            assert num_class is not None, "classification 任务需要 num_class"
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout_layer = nn.Dropout(dropout)
            self.projection = nn.Linear(self.head_nf * enc_in, num_class)

        # 是否做 enc_in -> c_out 的线性转换
        if self.trans_enc_in_to_c_out:
            self.linear = SeriesfitLinear(
                linear_in=enc_in,
                w_len=sf_w_len,
                linear_out=self.c_out,
                series_fit_add_res=sf_series_fit_add_res,
                moving_avg=sf_moving_avg,
                linear_dropout=sf_linear_dropout,
            )

    # ========= 各种任务 =========
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: [B, L, N]  (L = seq_len, N = enc_in)
        返回:  [B, pred_len, C_out] （默认 C_out = enc_in）
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()     # [B, 1, N]
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)                   # [B, N, L]
        # enc_out: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # enc_out: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # enc_out: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # enc_out: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder head
        dec_out = self.head(enc_out)                     # [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)               # [bs x target_window x nvars]

        # De-Normalization
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        # 现在 dec_out: [B, pred_len, enc_in]

        if self.trans_enc_in_to_c_out:
            # SeriesfitLinear 接受 [B, T, Cin]，输出 [B, N, Cout]
            dec_out = self.linear(dec_out)               # T -> N = T - w_len + 1

        return dec_out

    def imputation(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 原实现保留，不赘述，用的时候再说
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        return dec_out

    def classification(self, x_enc, x_mark_enc=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        output = self.flatten(enc_out)
        output = self.dropout_layer(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    # ========= DLinear 风格的 forward =========
    def forward(self, x):
        """
        DLinear 接口：
          x: [B, seq_len, enc_in]
        默认执行 long_term_forecast / short_term_forecast 路径，
        返回: [B, pred_len, enc_out]（若没启用 trans_enc_in_to_c_out，则 enc_out = enc_in）
        """
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
            # 保留原代码里的切片形式（一般没影响）
            if dec_out.size(1) >= self.pred_len:
                dec_out = dec_out[:, -self.pred_len:, :]
            return dec_out

        elif self.task_name == "imputation":
            return self.imputation(x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None)

        elif self.task_name == "anomaly_detection":
            return self.anomaly_detection(x)

        elif self.task_name == "classification":
            return self.classification(x, x_mark_enc=None)

        # 默认当作预测
        dec_out = self.forecast(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
        if dec_out.size(1) >= self.pred_len:
            dec_out = dec_out[:, -self.pred_len:, :]
        return dec_out
