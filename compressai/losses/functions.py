# --- START OF FILE functions.py (Corrected) ---

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .percentile import Percentile
import cv2
import numpy as np


def upsample(img, odd, filt):
    img = F.pad(img, (1, 1, 1, 1), mode='replicate')
    h = 2 * img.shape[2]
    w = 2 * img.shape[3]
    # Create tensor on the same device as input
    o = torch.zeros([img.shape[0], img.shape[1], h, w], device=img.device)
    o[:, :, 0:h:2, 0:w:2] = 4 * img
    o = F.conv2d(o, filt, padding=math.floor(filt.shape[2] / 2))
    o = o[:, :, 2:h - 2 - odd[0], 2:w - 2 - odd[1]]

    return o


def downsample(img, filt):
    pad = math.floor(filt.shape[2]/2)
    img = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    o = F.conv2d(img, filt)
    o = o[:, :, :img.shape[2]:2, :img.shape[3]:2]

    return o


def laplacian_pyramid_s(img, n_lev, filt):
    pyr = [0] * n_lev  # [0, 0, 0, ...]
    o = img

    for i in range(0, n_lev - 1):
        g = downsample(o, filt)
        h_odd = g.shape[2] * 2 - o.shape[2]
        w_odd = g.shape[3] * 2 - o.shape[3]
        pyr[i] = o - upsample(g, [h_odd, w_odd], filt)
        o = g

    pyr[n_lev - 1] = o

    return pyr


def nlp(img, n_lev, params):  # 求得原图的拉普拉斯金字塔
        npyr = [0] * n_lev
        img = torch.pow(img, 1 / params['gamma'])
        pyr = laplacian_pyramid_s(img, n_lev, params['F1'])

        for i in range(0, n_lev-1):
            pad = math.floor(params['filts'][0].shape[2] / 2)
            apyr = F.pad(torch.abs(pyr[i]), (pad, pad, pad, pad), mode='replicate')
            den = F.conv2d(apyr, params['filts'][0]) + params['sigmas'][0]
            npyr[i] = pyr[i] / den

        pad = math.floor(params['filts'][1].shape[2] / 2)
        apyr = F.pad(torch.abs(pyr[n_lev-1]), (pad, pad, pad, pad), mode='replicate')
        den = F.conv2d(apyr, params['filts'][1]) + params['sigmas'][1]

        npyr[n_lev-1] = pyr[n_lev-1] / den

        return npyr


class NLPD_Loss(torch.nn.Module):
    def __init__(self):
        super(NLPD_Loss, self).__init__()
        # Store original parameters. They will be moved to the correct device in forward().
        self.params = dict()
        self.params['gamma'] = 2.60
        self.params['filts'] = dict()
        self.params['filts'][0] = torch.tensor([[0.0400, 0.0400, 0.0500, 0.0400, 0.0400],
                                                [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                                [0.0500, 0.0400, 0.0500, 0.0400, 0.0500],
                                                [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                                [0.0400, 0.0400, 0.0500, 0.0400, 0.0400]],
                                                dtype=torch.float).unsqueeze(0).unsqueeze(0)

        self.params['filts'][1] = torch.tensor([[0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0]],
                                                dtype=torch.float).unsqueeze(0).unsqueeze(0)

        self.params['sigmas'] = torch.tensor([0.1700, 4.8600], dtype=torch.float)

        self.params['F1'] = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                                          [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                          [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                                          [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                          [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                                          dtype=torch.float).unsqueeze(0).unsqueeze(0)

        self.exp_s = 2.00
        self.exp_f = 0.60

    def forward(self, h_img, l_img, n_lev=None):
        ldr_min = 5.0
        ldr_max = 300.0
        cali_ldr = (ldr_max - ldr_min) * l_img + ldr_min
        l_img = cali_ldr

        if n_lev is None:
            n_lev = math.floor(math.log(min(h_img.shape[2:]), 2)) - 2

        # Create a temporary params dict for this forward pass to ensure statelessness
        device = h_img.device
        params = {
            'gamma': self.params['gamma'],
            'sigmas': self.params['sigmas'].to(device),
            'F1': self.params['F1'].to(device),
            'filts': {
                0: self.params['filts'][0].to(device),
                1: self.params['filts'][1].to(device)
            }
        }
        
        h_pyr = nlp(h_img, n_lev, params)
        l_pyr = nlp(l_img, n_lev, params)

        dis = []

        for i in range(0, n_lev):
            diff = torch.pow(torch.abs(h_pyr[i] - l_pyr[i]), self.exp_s)
            diff_pow = torch.pow(torch.mean(torch.mean(diff, dim=-1), dim=-1), self.exp_f / self.exp_s)
            dis.append(diff_pow)

        dis = torch.cat(dis, -1)
        loss = torch.pow(torch.mean(dis, dim=-1), 1. / self.exp_f)

        return loss.mean()


class LDR_Seq(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq, self).__init__()

    def get_luminance(self,img):
        if (img.shape[1] == 3):
            R = img[:, 2, :, :]
            G = img[:, 1, :, :]
            B = img[:, 0, :, :]
            Y = R * 0.212656 + G * 0.715158 + B * 0.072186
        elif (img.shape[1] == 1):
            Y = img
        else:
            # This should raise an error, not just print
            raise ValueError('Error: get_luminance: wrong matrix dimension')
        return Y

    def generation(self, img):
       b = 0
    L = self.get_luminance(img)
    img_l = torch.log2(L + 0.5)
    
    # --- TEMPORARY DEBUGGING ---
    # l_img = Percentile()(img_l.reshape(1, -1).squeeze(), [0, 100])
    
    # Replace percentile with a simple min/max to test if Percentile is the issue
    # This is a good approximation of the 0th and 100th percentile
    print("[DEBUG] Using torch.min/max instead of Percentile")
    l_min_val = torch.min(img_l)
    l_max_val = torch.max(img_l)
    # --- END OF DEBUGGING ---

    # Use the new values
    l_min = l_min_val
    l_max = l_max_val

    f8_stops = torch.ceil((l_max - l_min) / 8)
        l_start = l_min + (l_max - l_min - f8_stops * 8) / 2
        
        # --- CRITICAL FIX FOR TPU HANG ---
        # Calculate 'number' as a tensor first to preserve the computation graph.
        number_tensor = 3 * f8_stops
        # Then, get its Python integer value to use in the loop. .item() implicitly moves to CPU.
        number = int(number_tensor.item())
        # --- END OF FIX ---

        result = []
        ek_value = []
        for i in range(number):
            k = i * 8 + 3
            ek = 2 ** (l_start + (k / 3))
            img1 = (img / (ek + 1e-8) - b) / (1 - b)
            imgClamp = img1.clamp(1e-8, 1)
            imgP = (imgClamp) ** (1 / 2.2)

            result.append(imgP)
            ek_value.append(ek)
        return result, ek_value


class LDR_Seq_out(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq_out, self).__init__()

    def generation(self, img, ek_value):
        b = 0
        number = len(ek_value)

        result = []
        for i in range(number):
            ek = ek_value[i]
            img1 = (img / (ek + 1e-8) - b) / (1 - b)
            imgClamp = img1.clamp(1e-8, 1)
            imgP = (imgClamp) ** (1 / 2.2)

            result.append(imgP)
        return result


class hdrMetric(torch.nn.Module):
    def __init__(self):
        super(hdrMetric, self).__init__()
        self.generate_GT = LDR_Seq()
        self.generate_out = LDR_Seq_out()
        self.loss_fun = nn.L1Loss()

    def forward(self, output, gt):
        # --- CRITICAL FIX FOR DYNAMIC GRAPH ---
        # All calculations depending on the ground truth (gt) should NOT be part of the
        # computation graph for the backward pass of the main model.
        # Use torch.no_grad() to compute the exposure values (ek) and the
        # ground truth sequence (gt_seq) as fixed targets.
        
        with torch.no_grad():
            gt_seq, ek = self.generate_GT.generation(gt)

        # Now, only the generation of the output sequence is part of the autograd graph.
        # The `ek` values are now treated as constants.
        # The loop inside generate_out is now of a fixed length for this forward pass.
        output_seq = self.generate_out.generation(output, ek)

        # The loss calculation remains the same.
        Q = []
        for k in range(len(output_seq)):
            # The target, gt_seq[k], does not have a grad_fn.
            # The input, output_seq[k], does. This is correct.
            Qk = self.loss_fun(gt_seq[k], output_seq[k])
            Q.append(Qk)

        loss = torch.sum(torch.stack(Q))
        return loss

# The rest of the file from here down seems okay for now as they are mostly helper functions
# or use in-place operations that might not be in the training autograd path (e.g. BGR_HSV).
# If you run into further issues, the BGR_HSV class would be the next place to refactor.

def num_to_string(num):
    numbers = {
        'banding': [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484],
        'banding_glare': [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142],
        'peaks': [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577],
        'peaks_glare': [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374]
    }
    return numbers.get(num, None)

def PU21_encoding(Y):
    L_min = 0.005
    L_max = 10000
    Y = torch.clip(Y, L_min, L_max)
    p = num_to_string('banding_glare')
    value = p[6] * (((p[0] + p[1] * Y ** p[3]) / (1 + p[2] * Y ** p[3])) ** p[4] - p[5])
    V = torch.clip(value, 0, 1e16)
    return V

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 1) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def color_reproduce(ldr, ref_hdr, hsv_ldr_hat, hsv_target_hdr):
    v_hdr = hsv_target_hdr[:, 2, :, :]
    v_ldr = hsv_ldr_hat[:, 2, :, :]
    ldr[:, 2, :, :] = torch.pow(ref_hdr[:, 2, :, :]/v_hdr, 0.6) * v_ldr
    ldr[:, 1, :, :] = torch.pow(ref_hdr[:, 1, :, :] / v_hdr, 0.6) * v_ldr
    ldr[:, 0, :, :] = torch.pow(ref_hdr[:, 0, :, :] / v_hdr, 0.6) * v_ldr
    return ldr

class BGR_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(BGR_HSV, self).__init__()
        self.eps = eps

    def forward(self, img):
        permute = [2, 1, 0]
        img = img[:, permute, :, :]
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 0] == img.max(1)[0]]) % 6
        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6
        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0
        value = img.max(1)[0]
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_bgr(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))
        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        bgr = torch.cat([b, g, r], dim=1)
        return bgr

# --- END OF FILE ---
