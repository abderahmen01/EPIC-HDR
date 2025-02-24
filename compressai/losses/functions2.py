import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .percentile import Percentile

class LDR_Seq(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq, self).__init__()

    def get_luminance(self,img):
        # print('img.shape: ', img.shape)
        if (img.shape[1] == 3):
            # R = img[:, 2, :, :]
            # G = img[:, 1, :, :]
            # B = img[:, 0, :, :]
            Y = img[:, 2, :, :]
            # cv2.imread --> BGR
        elif (img.shape[1] == 1):
            Y = img
        else:
            print('Error: get_luminance: wrong matrix dimension')
        return Y

    def generation(self, img):

        #img_q = img[img >= 0]
        b = 0#1 / 128
        #min_v = torch.min(img_q)
        #img[img < 0] = min_v
        L = self.get_luminance(img)
        img_l = torch.log2(L+0.5)
        l_img = Percentile()(img_l[:].reshape(1, -1).squeeze(), [0, 100])
        l_min = l_img[0]
        l_max = l_img[1]
        # l_min = l_min
        f8_stops = torch.ceil((l_max - l_min) / 8)
        l_start = l_min + (l_max - l_min - f8_stops * 8) / 2
        number = 8 * 3 * f8_stops / 8
        number = torch.tensor((number), dtype=torch.int64)

        result = []
        ek_value = []
        for i in range(number):
            k = i * 8 + 4
            ek = 2 ** (l_start + ((k / 3)))
            img1 = (img / (ek+0.00000001) - b) / (1 - b)
            imgClamp = img1.clamp(1e-12, 1)#torch.clamp(img1,0, 1)#torch.clip,torch.sigmoid(img1)#
            imgP = (imgClamp) ** (1 / 2.2)

            # file_name = '%d.png' % k
            # wfid1 = os.path.join('./result_pytorch/', file_name)
            # plt.imsave(wfid1, imgP.squeeze().permute(1, 2, 0).numpy())
            result.append(imgP)
            ek_value.append(ek)
        return result, ek_value


class LDR_Seq_out(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq_out, self).__init__()

    def generation(self, img, ek_value):

        #img_q = img[img >= 0]
        b = 0#1 / 128
        #min_v = torch.min(img_q)
        #img[img < 0] = min_v
        number = len(ek_value)


        result = []
        for i in range(number):
            ek = ek_value[i]
            img1 = (img / (ek+0.00000001) - b) / (1 - b)
            imgClamp = img1.clamp(1e-12, 1)  #torch.clamp(img1,0, 1)#, 0, 1torch.sigmoid(img1) #
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
        gt_seq, ek = self.generate_GT.generation(gt)
        output_seq = self.generate_out.generation(output, ek)

        Q = []
        for k in range(len(output_seq)):
            Qk = self.loss_fun(gt_seq[k], output_seq[k])
            Q.append(Qk)

        loss = torch.sum(torch.stack(Q))
        return loss


def num_to_string(num):
    numbers = {
        'banding': [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484],
        'banding_glare': [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142],
        'peaks': [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577],
        'peaks_glare': [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374]
    }

    return numbers.get(num, None)


def PU21_encoding(Y):
    # epsilon = 1e-5
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
    ldr[:, 2, :, :] = torch.pow(ref_hdr[:, 2, :, :]/v_hdr, 0.6) * v_ldr  # r_ldr
    ldr[:, 1, :, :] = torch.pow(ref_hdr[:, 1, :, :] / v_hdr, 0.6) * v_ldr  # g_ldr
    ldr[:, 0, :, :] = torch.pow(ref_hdr[:, 0, :, :] / v_hdr, 0.6) * v_ldr  # b_ldr

    return ldr


class BGR_HSV(nn.Module):
    """
    Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
    RGB or HSV's shape: (B * C * H * W)
    RGB or HSV's range: [0, 1)
    """
    def __init__(self, eps=1e-8):
        super(BGR_HSV, self).__init__()
        self.eps = eps

    def forward(self, img):

        # bgr to rgb
        permute = [2, 1, 0]
        img = img[:, permute, :, :]

        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

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
        # 对出界值的处理
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