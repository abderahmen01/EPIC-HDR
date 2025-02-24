# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import skimage.color as color
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model
import math
from .base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from .utils import conv, deconv

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    conv1x1,
    subpel_conv3x3,
    ResBlock4hup,
    CondResidualBlock,
)

from .pconvunet import *

__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
    "ldr2hdr",
    "end2end",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",

]


@register_model("bmshj2018-factorized")
class FactorizedPrior(CompressionModel):

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

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

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
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


@register_model("bmshj2018-factorized-relu")
class FactorizedPriorReLU(FactorizedPrior):

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, 3),
        )


@register_model("bmshj2018-hyperprior")
class ScaleHyperprior(CompressionModel):

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

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

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("mbt2018-mean")
class MeanScaleHyperprior(ScaleHyperprior):

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules such as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """
    def forward(self, x, t_emb):
        for layer in self:
            if isinstance(layer, CondResidualBlock):
                x = layer(x, t_emb)
            # elif isinstance(layer, SpatialTransformer):
            #     x = layer(x, cond)
            else:
                x = layer(x)
        return x

@register_model("mbt2018")
class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
            AttentionBlock(M),
        )

        self.d_t_emb = 4 * N

        self.g_s_ldr = TimestepEmbedSequential(
            AttentionBlock(M),
            CondResidualBlock(M, N, self.d_t_emb),
            ResidualBlockUpsample(N, N, 2),
            CondResidualBlock(N, N, self.d_t_emb),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            CondResidualBlock(N, N, self.d_t_emb),
            ResidualBlockUpsample(N, N, 2),
            CondResidualBlock(N, N, self.d_t_emb),
            subpel_conv3x3(N, 3, 2),
        )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, M, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(M, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, M * 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(N, 4 * N),
            nn.SiLU(),
            nn.Linear(4 * N, 4 * N),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.N = int(N)
        self.M = int(M)
        self.channels = self.N
        print("N=", self.N)
        print("M=", self.M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, smax):
        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        t_emb = self.time_step_embedding(smax)
        t_emb = self.time_embed(t_emb)
        hh = self.g_s_ldr(y_hat, t_emb)
        ldr_x_hat = F.sigmoid(hh)

        return {
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "ldr_x_hat": ldr_x_hat
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = 192
        M = 192
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)
        hyper_z = self.hyper_h_a(z)

        hyper_z_strings = self.hyper_entropy_bottleneck.compress(hyper_z)
        hyper_z_hat = self.hyper_entropy_bottleneck.decompress(hyper_z_strings, hyper_z.size()[-2:])

        hyper_params = self.hyper_h_s(hyper_z_hat)

        s = 4  # scaling factor between  hyper_z and z
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        z_height = hyper_z_hat.size(2) * s
        z_width = hyper_z_hat.size(3) * s

        z_hat = F.pad(z, (padding, padding, padding, padding))

        z_strings = []
        for i in range(z.size(0)):
            string = self._compress_ar(
                z_hat[i: i + 1],
                hyper_params[i: i + 1],
                z_height,
                z_width,
                kernel_size,
                padding,
            )
            z_strings.append(string)

        z_hat = torch.zeros(
            (hyper_z_hat.size(0), self.M//8, z_height + 2 * padding, z_width + 2 * padding),
            device=hyper_z_hat.device,
        )

        for i, z_string in enumerate(z_strings):
            self._decompress_ar(
                z_string,
                z_hat[i: i + 1],
                hyper_params[i: i + 1],
                z_height,
                z_width,
                kernel_size,
                padding,
            )

        z_hat = F.pad(z_hat, (-padding, -padding, -padding, -padding))

        y_base = self.h_up_sample(z_hat)
        y_res = y - y_base

        ### Visualize y_base and y_residual
        # for i in range(y_base[0].size(0)):
        #     torchvision.utils.save_image(y_base[0][i], './y_base/' + str(i) + '_chs_y_base.png')
        #
        # for i in range(y_res[0].size(0)):
        #     torchvision.utils.save_image(y_res[0][i], './y_res/' + str(i) + '_chs_y_res.png')

        y_strings = self.entropy_bottleneck_res.compress(y_res)

        return {"strings": [y_strings, z_strings, hyper_z_strings], "y_res_shape": y_res.size()[-2:], "hyper_z_shape": hyper_z.size()[-2:]}

        # return {"strings": [z_strings, hyper_z_strings], "hyper_z_shape": hyper_z.size()[-2:]}


    def _compress_ar(self, z_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.hyper_context_prediction.weight * self.hyper_context_prediction.mask
        for h in range(height):
            for w in range(width):
                z_crop = z_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    z_crop,
                    masked_weight,
                    bias=self.hyper_context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.hyper_entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.hyper_gaussian_conditional.build_indexes(scales_hat)

                z_crop = z_crop[:, :, padding, padding]
                z_q = self.hyper_gaussian_conditional.quantize(z_crop, "symbols", means_hat)
                z_hat[:, :, h + padding, w + padding] = z_q + means_hat

                symbols_list.extend(z_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string


    def decompress(self, strings, hyper_z_shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        hyper_z_hat = self.hyper_entropy_bottleneck.decompress(strings[1], hyper_z_shape)
        hyper_params = self.hyper_h_s(hyper_z_hat)

        s = 4  # scaling factor between hyper_z and z
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        z_height = hyper_z_hat.size(2) * s
        z_width = hyper_z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        z_hat = torch.zeros(
            (hyper_z_hat.size(0), self.M//8, z_height + 2 * padding, z_width + 2 * padding),
            device=hyper_z_hat.device,
        )


        for i, z_string in enumerate(strings[0]):
            self._decompress_ar(
                z_string,
                z_hat[i: i + 1],
                hyper_params[i: i + 1],
                z_height,
                z_width,
                kernel_size,
                padding,
            )

        z_hat = F.pad(z_hat, (-padding, -padding, -padding, -padding))

        y_base = self.h_up_sample(z_hat)

        y_res_hat = self.entropy_bottleneck_res.decompress(strings[0], y_shape)

        y_hat = y_base + y_res_hat
        # x_hat = torch.clamp(self.g_s(y_hat), 1e-8, 1)
        #
        # ldr_x_hat = torch.clamp(self.g_s_ldr(y_base), 1e-8, 1)
        x_hat = F.sigmoid(self.g_s(y_hat))

        ldr_x_hat = F.sigmoid(self.g_s_ldr(y_base))

        return {"x_hat": x_hat, "ldr_x_hat": ldr_x_hat}

        # return {"ldr_x_hat": ldr_x_hat}


    def _decompress_ar(
        self, z_string, z_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.hyper_gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.hyper_gaussian_conditional.cdf_length.tolist()
        offsets = self.hyper_gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(z_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                z_crop = z_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    z_crop,
                    self.hyper_context_prediction.weight,
                    bias=self.hyper_context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.hyper_entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.hyper_gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.hyper_gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                z_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


@register_model("ldr2hdr")
class ldr2hdr(MeanScaleHyperprior):

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.unet_enc_hdr = PConvUNet_Enc_hdr()
        self.unet_dec_hdr = PConvUNet_Dec_hdr()
        self.unet_enc_ldr = PConvUNet_Enc_ldr()
        self.unet_dec = PConvUNet_Dec()

        self.entropy_bottleneck = EntropyBottleneck(512)

        print("N=", self.N)
        print("M=", self.M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, hdr, ldr):
        h_hdr, h_dict = self.unet_enc_hdr(hdr)

        h_hdr_hat, h_hdr_likelihoods = self.entropy_bottleneck(h_hdr)

        _, h_dict_hdr = self.unet_dec_hdr(h_hdr_hat, h_dict)

        h_ldr, h_dict_ldr = self.unet_enc_ldr(ldr, h_dict_hdr)

        hdr_recon = F.sigmoid(self.unet_dec(h_ldr, h_dict_ldr, h_dict_hdr))

        return {
            "likelihoods": {"h_hdr": h_hdr_likelihoods},
            "hdr": hdr_recon
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = 192
        M = 192
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


@register_model("end2end")
class end2end(nn.Module):
    def __init__(self, modelA, modelB):
        super(end2end, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x, smax, hdr):
        out_net1 = self.modelA(x, smax)
        ldr_hat = out_net1["ldr_x_hat"]
        ldr_hat_v = ldr_hat[:, 2, :, :].unsqueeze(1)
        ldr_hat_hs = ldr_hat[:, 0:2, :, :]

        ldr_out_v2 = (300 - 5) * ldr_hat_v + 5
        ldr_out2 = torch.cat([ldr_hat_hs, ldr_out_v2], dim=1)
        # print('ldr_out2.shape=', ldr_out2.shape)

        d_max = 300
        d_min = 5
        hdr_h = ldr_out2.data[0] #.permute(1, 2, 0)
        # print('hdr_h.shape: ', hdr_h.shape)
        t = hdr_h[2, :, :]  # .cpu()
        # print('t.shape: ', t.shape)
        # print('t.squeeze().shape: ', t.squeeze().shape)
        # t[t > d_max] = d_max
        # t[t < d_min] = d_min
        t = torch.clamp(t, min=d_min, max=d_max)
        t = (t - d_min) / (d_max - d_min)
        t = (t ** (1 / 2.2))
        hdr_h = hdr_h  # .cpu().numpy()
        hdr_h[2, :, :] = t #.squeeze().cpu().numpy()
        hdr_h[1, :, :] = hdr_h[1, :, :] * 0.6

        hdr_h = hdr_h.unsqueeze(dim=0)
        # print('hdr_h.shape: ', hdr_h.shape)
        ldr = hsv2rgb_torch(hdr_h)

        out_net2 = self.modelB(hdr, ldr)

        return out_net1, out_net2

