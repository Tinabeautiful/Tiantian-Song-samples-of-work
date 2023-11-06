import math
import warnings

import torch.nn as nn

from entropy_model import EntropyBottleneck
from com_utils import conv, deconv, update_registered_buffers
from filters import FilterLow, FilterHigh
from layers import *

class CompressionModel(nn.Module):

    def __init__(self, entropy_bottleneck_channels, init_weights=None):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights is not None:
            warnings.warn(
                "init_weights was removed as it was never functional",
                DeprecationWarning,
            )

    def aux_loss(self):
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

class FC_fullyfactorized(CompressionModel):

    def __init__(self, N=128, M=192, **kwargs):
        super().__init__(entropy_bottleneck_channels=M,  **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M
        self.filter_low_module = FilterLow(kernel_size=5, gaussian=True, include_pad=False)
        self.filter_high_module =FilterHigh(kernel_size=5, gaussian=True, include_pad=False, normalize=False)

    def forward(self, x):
        y = self.g_a(x)
        assert not math.isinf(y[0][0][0][0])
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        x_LFC = self.filter_low_module(x)
        x_hat_LFC = self.filter_low_module(x_hat)
        x_hat_HFC = self.filter_high_module(x_hat)

        return {  
            "x":x,
            "x_hat": x_hat,
            "x_LFC": x_LFC,
            "x_HFC_hat":x_hat_HFC,
            "x_LFC_hat":x_hat_LFC,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class FC_fully_residual(FC_fullyfactorized):

    def __init__(self, N=128, M=128, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.conv1.weight"].size(0)
        M = N
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net