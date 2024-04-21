from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpeechEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 40,
        cnn_dim: List[int] = [128, 512],
        kernel_size: int = 6,
        stride: int = 1,
        rnn_dim: int = 512,
        rnn_num_layers: int = 2,
        rnn_type: str = "gru",
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = True,
        attn_heads: int = 1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        assert rnn_type in ["lstm", "gru"]
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim[0], kernel_size, stride),
            nn.BatchNorm1d(cnn_dim[0]),
            nn.ReLU(), #new
            nn.Conv1d(cnn_dim[0], cnn_dim[1], kernel_size-3, stride),
            nn.BatchNorm1d(cnn_dim[1]),
            nn.ReLU(), #new
        )
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.rnn_bidirectional = rnn_bidirectional
        rnn_kwargs = dict(
            input_size=cnn_dim[1],
            hidden_size=rnn_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            self.rnn = nn.GRU(**rnn_kwargs)
        self.output_dim = rnn_dim * (int(rnn_bidirectional) + 1)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # thêm fix forward
        self.linear = nn.Linear(self.output_dim, 512)

        self.linear2 = nn.Linear(512, 256)

        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(256, 128)

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, mel_spec, mel_spec_len):
        """
        mel_spec (-1, 40, len)
        output (-1, len, rnn_dim * (int(bidirectional) + 1))
        """
        cnn_out = self.cnn(mel_spec)

        # l = [
        #     torch.div(y - self.kernel_size, self.stride, rounding_mode="trunc") + 1
        #     for y in mel_spec_len
        # ]
        # l = [
        #     torch.div(y - self.kernel_size, self.stride, rounding_mode="trunc") + 1
        #     for y in l
        # ]

        cnn_out = cnn_out.permute(0, 2, 1)

        # packed = pack_padded_sequence(
        #     cnn_out, l, batch_first=True, enforce_sorted=False
        # )
        # self.rnn.flatten_parameters()
        # out, hidden_state = self.rnn(packed)
        # out, seq_len = pad_packed_sequence(out, batch_first=True)
        # pack input before RNN to reduce computing efforts
        out, hidden_state = self.rnn(cnn_out)

        out, weights = self.self_attention(out, out, out)
        out = out.mean(dim=1)  # mean the time step

        print(out.shape)
        out = self.linear(out)
        print(out.shape)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)

        out = torch.nn.functional.normalize(out)
        return out
