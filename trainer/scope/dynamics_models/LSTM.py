from typing import Optional

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    DLinear 接口风格的 LSTM 时间序列模型：
      __init__(seq_len, pred_len, individual, enc_in, ...)
      forward(x): x [B, seq_len, enc_in] -> [B, pred_len, c_out]

    设计思路：
      1) 用一个（或多层）LSTM 把输入序列编码成最后时刻的 hidden state
      2) 用一个全连接层把 hidden state 映射到 pred_len * c_out
      3) reshape 到 [B, pred_len, c_out]
      4) 可选的按时间维做归一化 / 反归一化（和前面 Transformer/TST/TimesNet 类似）
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        individual: bool,
        enc_in: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        c_out: Optional[int] = None,
        dropout: float = 0.0,
        use_norm: bool = False,
    ):
        """
        参数:
          seq_len   : 输入序列长度
          pred_len  : 预测长度
          individual: 与 DLinear 接口对齐，占位用（当前实现没用到）
          enc_in    : 输入通道数（特征维数）
          hidden_size : LSTM 的隐藏维度
          num_layers  : LSTM 堆叠层数
          bidirectional : 是否双向 LSTM
          c_out      : 输出特征维数，默认等于 enc_in
          dropout    : LSTM 层之间的 dropout（num_layers > 1 时生效）
          use_norm   : 是否对输入做 per-sequence 标准化并在输出端反标准化
        """
        super().__init__()

        # ---- DLinear 对齐的核心参数 ----
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
        self.c_out = c_out if c_out is not None else enc_in
        self.use_norm = use_norm

        # ---- LSTM 编码器 ----
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # 输入 [B, L, C]
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 双向的话，hidden 维度要乘 2
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        # ---- 把最后一个 hidden 映射到 pred_len * c_out ----
        self.fc = nn.Linear(lstm_out_dim, pred_len * self.c_out)

    def forward(self, x):
        """
        x: [B, seq_len, enc_in]
        返回:
          y: [B, pred_len, c_out]
        """
        # -------- 可选的标准化（按时间维度对每个 batch & 通道做） --------
        if self.use_norm:
            # means, std: [B, 1, C]
            means = x.mean(dim=1, keepdim=True).detach()
            x_centered = x - means
            stdev = torch.sqrt(
                torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_norm = x_centered / stdev
        else:
            x_norm = x
            means = None
            stdev = None

        B = x_norm.size(0)

        # -------- LSTM 编码 --------
        # outputs: [B, L, lstm_out_dim]
        # h_n: [num_layers * num_directions, B, hidden_size]
        outputs, (h_n, c_n) = self.lstm(x_norm)

        # 取最后一层的 hidden state
        # 如果是多层 + 双向：h_n 形状 [num_layers * num_directions, B, hidden_size]
        if self.bidirectional:
            # 拆出最后一层的正向和反向 hidden 并拼接
            # 维度索引： (num_layers-1)*2 是最后一层正向，+1 是最后一层反向
            h_last_forward = h_n[-2, :, :]   # [B, hidden_size]
            h_last_backward = h_n[-1, :, :]  # [B, hidden_size]
            h_last = torch.cat([h_last_forward, h_last_backward], dim=-1)  # [B, 2*hidden_size]
        else:
            h_last = h_n[-1, :, :]           # [B, hidden_size]

        # -------- 映射到未来 pred_len 序列 --------
        out = self.fc(h_last)                # [B, pred_len * c_out]
        out = out.view(B, self.pred_len, self.c_out)  # [B, pred_len, c_out]

        # -------- 反标准化 --------
        if self.use_norm:
            # means, stdev: [B, 1, enc_in] -> 需要广播 / 截断到 c_out 维度
            # 如果 c_out != enc_in，就按前 c_out 维进行缩放
            if self.c_out <= self.enc_in:
                means_c = means[:, :, :self.c_out]
                stdev_c = stdev[:, :, :self.c_out]
            else:
                # c_out 比 enc_in 大的情况不太常见，这里简单做 repeat
                repeat_times = (self.c_out + self.enc_in - 1) // self.enc_in
                means_c = means.repeat(1, 1, repeat_times)[:, :, :self.c_out]
                stdev_c = stdev.repeat(1, 1, repeat_times)[:, :, :self.c_out]

            out = out * stdev_c + means_c    # [B, pred_len, c_out]

        return out
