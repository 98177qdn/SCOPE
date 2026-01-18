import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block with DFT
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # x: [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[:, 0, :] = 0  # 去掉直流分量
        # top-k 频率（按幅度）
        top_k_freq, _ = torch.topk(freq, self.top_k, dim=1)
        thresh = top_k_freq.min(dim=1, keepdim=True).values  # [B, 1, C]
        xf = xf * (freq > thresh)  # 只保留 top_k 频率
        x_season = torch.fft.irfft(xf, n=x.size(1), dim=1)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int):
        super(MultiScaleSeasonMixing, self).__init__()

        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        self.down_sampling_layers_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        """
        season_list: List of tensors with shapes [B, C, T_i]
        按“高->低”尺度混合
        """
        out_high = season_list[0]
        out_low = season_list[1] if len(season_list) > 1 else season_list[0]
        out_season_list = [out_high.permute(0, 2, 1)]  # [B, T, C]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers_modules[i](out_high)  # [B, C, T_low]
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))  # [B, T, C]

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int):
        super(MultiScaleTrendMixing, self).__init__()

        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        """
        trend_list: List of [B, C, T_i]
        按“低->高”尺度混合
        """
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1] if len(trend_list_reverse) > 1 else trend_list_reverse[0]
        out_trend_list = [out_low.permute(0, 2, 1)]  # [B, T, C]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff: int,
        down_sampling_window: int,
        down_sampling_layers: int,
        channel_independence: bool,
        decomp_method: str = "moving_avg",
        moving_avg: int = 25,
        top_k: int = 5,
        dropout: float = 0.1,
    ):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if decomp_method == "moving_avg":
            self.decompsition = series_decomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError("decompsition is error")

        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        """
        x_list: List of tensors [B, T_i, d_model]
        返回 out_list: List of [B, T_i, d_model]
        """
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if hasattr(self, "cross_layer"):
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))  # [B, C, T]
            trend_list.append(trend.permute(0, 2, 1))    # [B, C, T]

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)  # [B, T_i, C]
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)     # [B, T_i, C]

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend  # [B, T_i, C]
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):
    """
    多尺度 Past Decomposable Mixing 模型，改成 DLinear 接口：

      __init__(seq_len, pred_len, individual, enc_in, ...)
      forward(x): x [B, seq_len, enc_in] -> [B, pred_len, c_out]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        individual: bool,
        enc_in: int,
        # 主结构参数
        d_model: int = 64,
        d_ff: int = 256,
        e_layers: int = 2,
        down_sampling_layers: int = 1,
        down_sampling_window: int = 2,
        channel_independence: bool = False,
        # 分解与归一化
        decomp_method: str = "moving_avg",  # 'moving_avg' or 'dft_decomp'
        moving_avg: int = 25,
        top_k: int = 5,
        use_norm: int = 1,                  # 1: 使用 Normalize, 0: 关闭
        # 下采样
        down_sampling_method: str = "max",  # 'max', 'avg', 'conv', or 'none'
        # 嵌入
        embed: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
        # 输出维度
        c_out: int = None,
        # 任务类型（主要还是 long_term_forecast）
        task_name: str = "long_term_forecast",
        num_class: int = None,
        label_len: int = None,
    ):
        super(TimeMixer, self).__init__()

        # 与 DLinear 对齐的几个关键参数
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in

        self.task_name = task_name
        self.label_len = label_len if label_len is not None else seq_len
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.decomp_method = decomp_method
        self.moving_avg = moving_avg
        self.top_k = top_k
        self.use_norm = use_norm
        self.c_out = c_out if c_out is not None else enc_in

        # PDM blocks
        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    d_model=d_model,
                    d_ff=d_ff,
                    down_sampling_window=down_sampling_window,
                    down_sampling_layers=down_sampling_layers,
                    channel_independence=channel_independence,
                    decomp_method=decomp_method,
                    moving_avg=moving_avg,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )

        self.preprocess = series_decomp(moving_avg)

        # Embedding
        if channel_independence:
            # 每个通道单独嵌入
            self.enc_embedding = DataEmbedding_wo_pos(
                1, d_model, embed, freq, dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                enc_in, d_model, embed, freq, dropout
            )

        self.layer = e_layers

        # 多尺度 Normalize
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(
                    enc_in,
                    affine=True,
                    non_norm=True if use_norm == 0 else False,
                )
                for _ in range(down_sampling_layers + 1)
            ]
        )

        # === 预测任务相关的层 ===
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            # 每一尺度一个 predictor（时间维度从 T_i->pred_len）
            self.predict_layers = nn.ModuleList(
                [
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        pred_len,
                    )
                    for i in range(down_sampling_layers + 1)
                ]
            )

            if channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, self.c_out, bias=True)

                self.out_res_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            seq_len // (down_sampling_window ** i),
                            seq_len // (down_sampling_window ** i),
                        )
                        for i in range(down_sampling_layers + 1)
                    ]
                )

                self.regression_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            seq_len // (down_sampling_window ** i),
                            pred_len,
                        )
                        for i in range(down_sampling_layers + 1)
                    ]
                )

        # 其他任务的头（如果你以后需要再用）
        if self.task_name in ["imputation", "anomaly_detection"]:
            if channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, self.c_out, bias=True)

        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * seq_len, num_class)

    # ----------------- 下面基本沿用原逻辑 -----------------
    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)        # [B,T,c_out]
        out_res = out_res.permute(0, 2, 1)             # [B,C,T]
        out_res = self.out_res_layers[i](out_res)      # [B,C,T]
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)  # [B,pred_len,C]
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        """
        x_enc: [B, T, C]
        x_mark_enc: [B, T, D] or None
        返回:
          x_enc_list: List[[B, T_i, C]]
          x_mark_enc_list: List[[B, T_i, D]] or None
        """
        if self.down_sampling_method == "max":
            down_pool = nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == "avg":
            down_pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            # 不进行下采样
            return [x_enc], [x_mark_enc] if x_mark_enc is not None else None

        # 输入 [B, T, C] -> [B, C, T]
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []

        # 原始尺度
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))  # [B,T,C]
        if x_mark_enc is not None:
            x_mark_sampling_list.append(x_mark_enc)

        for _ in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)  # [B, C, T_down]
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))  # [B, T_down, C]
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling = x_mark_enc_ori[:, :: self.down_sampling_window, :]
                x_mark_sampling_list.append(x_mark_sampling)
                x_mark_enc_ori = x_mark_sampling

        x_enc_list = x_enc_sampling_list
        if x_mark_enc is not None:
            x_mark_list = x_mark_sampling_list
        else:
            x_mark_list = None

        return x_enc_list, x_mark_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [B, L, C]
        返回: [B, pred_len, c_out]
        """
        x_enc_list, x_mark_enc_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc_list is not None:
            for i, x, x_mark in zip(range(len(x_enc_list)), x_enc_list, x_mark_enc_list):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc_list)), x_enc_list):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_pre_list = self.pre_enc(x_list)
        if x_mark_enc_list is not None:
            for x, x_mark in zip(x_pre_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,d_model]
                enc_out_list.append(enc_out)
        else:
            for x in x_pre_list[0]:
                enc_out = self.enc_embedding(x, None)    # [B,T,d_model]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        B = x_enc.size(0)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_pre_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)  # [B,pred_len,c_out]
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            # x_list: (x_pre_list[0], x_pre_list[1]) 中的第一个
            x_list0 = x_list[0]
            N = self.enc_in
            real_B = B // N

            for i, enc_out in zip(range(len(x_list0)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # [B*,pred_len,d_model]
                dec_out = self.projection_layer(dec_out)  # [B*,pred_len,1]

                dec_out = dec_out.reshape(real_B, N, self.pred_len)  # [B,N,pred_len]
                dec_out = dec_out.permute(0, 2, 1).contiguous()      # [B,pred_len,N]
                dec_out_list.append(dec_out)
        else:
            # x_list: (out1_list, out2_list)
            out1_list, out2_list = x_list
            for i, enc_out, out_res in zip(
                range(len(out1_list)), enc_out_list, out2_list
            ):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # [B,pred_len,d_model]
                dec_out = self.out_projection(dec_out, i, out_res)  # [B,pred_len,c_out]
                dec_out_list.append(dec_out)

        return dec_out_list

    # 其他任务函数保留，略（你目前主要用 forecast）

    # ========== DLinear 风格的 forward ==========
    def forward(self, x):
        """
        DLinear 接口：
          x: [B, seq_len, enc_in]
        只做 long_term_forecast / short_term_forecast：
          返回: [B, pred_len, c_out]
        """
        dec_out = self.forecast(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
        return dec_out  # [B, pred_len, c_out]
