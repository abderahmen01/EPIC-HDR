from compressai.registry import register_criterion
from .functions import *


@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=200, beta=0.5, return_type="all"):
        super().__init__()
        self.hdr_metric = hdrMetric()
        self.ldr_metric = NLPD_Loss()
        self.beta = beta
        self.lmbda = lmbda
        self.return_type = return_type
        self.hs_fn = torch.nn.L1Loss()

    def forward(self, output1, output2, target_hsv, target, smax):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        smin = 5

        out["bpp_loss1"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output1["likelihoods"].values()
        )

        target_hdr = target_hsv
        ldr_hat = output1["ldr_x_hat"]
        ldr_hat_v = ldr_hat[:, 2, :, :].unsqueeze(1)
        ldr_hat_v = ((ldr_hat_v - ldr_hat_v.min()) / (ldr_hat_v.max() - ldr_hat_v.min()))
        ldr_hat_hs = ldr_hat[:, 0:2, :, :]

        target_hdr_v = target_hdr[:, 2, :, :].unsqueeze(1)
        target_hdr_hs = target_hdr[:, 0:2, :, :]

        target_hdr_v2 = ((target_hdr_v - target_hdr_v.min()) / (target_hdr_v.max() - target_hdr_v.min()))
        target_hdr_v2 = (smax - smin) * target_hdr_v2 + smin
        out["ldr_loss"] = self.ldr_metric(target_hdr_v2, ldr_hat_v) + 5 * self.hs_fn(ldr_hat_hs,target_hdr_hs)

        distortion1 = out["ldr_loss"]

        ###############################################
        # HDR reconstruction loss
        ###############################################
        out["bpp_loss2"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output2["likelihoods"].values()
        )

        target_hdr = target
        hdr_recon = output2["hdr"]
        hdr_recon_max = torch.max(torch.max(torch.max(hdr_recon, 1)[0], 1)[0], 1)[0].unsqueeze(1).unsqueeze(
            1).unsqueeze(1)
        # hdr_recon = (hdr_recon - hdr_recon.min()) / (hdr_recon.max() - hdr_recon.min()+ 1e-30)
        hdr_recon = (hdr_recon) / (hdr_recon_max + 1e-30)
        target_hdr_max = torch.max(torch.max(torch.max(target_hdr, 1)[0], 1)[0], 1)[0].unsqueeze(1).unsqueeze(
            1).unsqueeze(1)
        target_hdr = (target_hdr) / (target_hdr_max + 1e-30)
        # target_hdr = (target_hdr - target_hdr.min()) / (target_hdr.max() - target_hdr.min() + 1e-30)

        out["hdr_loss"] = self.hdr_metric(hdr_recon, target_hdr)

        distortion2 = out["hdr_loss"]

        out["loss"] = self.lmbda * distortion1 + 10 * out["bpp_loss1"] + self.lmbda * distortion2 + out["bpp_loss2"]

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
