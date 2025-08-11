import torch
import torch.nn as nn

class BrainNetMLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop_rate, norm, k):
        super().__init__()
        """
        dim: a tuple of the input dimension and output dimension
        hidden_dim: a tuple indicates the embedding dimension of spatial and spectral features
        drop_rate: a tuple of dropout rates for spatial and spectral projection
        norm: bool variable to determine whether batchnorm used or not
        k: the width of low-pass filter
        """
        in_dim, out_dim = dim
        self.rows, self.cols = torch.triu_indices(in_dim, in_dim, offset=0)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim[0] + hidden_dim[1], out_dim),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.Dropout(drop_rate[0]),

        )
        self.spatial_proj = nn.Sequential(
            nn.Linear((in_dim+1)*(in_dim//2), hidden_dim[1]),
            nn.Dropout(drop_rate[1]),
        )
        self.k = k
        if norm:
            self.norm = nn.BatchNorm1d(hidden_dim[0] + hidden_dim[1])
        else:
            self.norm = nn.Identity()
    def forward(self, x, ts):
        x_f = torch.fft.rfft(ts, dim=1)
        f_t = torch.abs(x_f)
        f_t = f_t[:, 1:self.k, :]
        f_t = self.freq_proj(f_t).mean(1)
        x = x[:, self.rows, self.cols]
        x = x.flatten(start_dim=1)
        x = self.spatial_proj(x)
        x = torch.cat((x, f_t), dim=1)
        x = self.norm(x)
        return self.decoder(x)
