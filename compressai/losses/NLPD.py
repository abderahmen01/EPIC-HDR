import torch
import math
import torch.nn.functional as F

# ... (upsample, downsample, laplacian_pyramid_s, nlp functions are OK) ...

def upsample(img, odd, filt):
    img = F.pad(img, (1, 1, 1, 1), mode='replicate')
    h = 2 * img.shape[2]
    w = 2 * img.shape[3]
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
    pyr = [0] * n_lev
    o = img
    for i in range(0, n_lev - 1):
        g = downsample(o, filt)
        h_odd = g.shape[2] * 2 - o.shape[2]
        w_odd = g.shape[3] * 2 - o.shape[3]
        pyr[i] = o - upsample(g, [h_odd, w_odd], filt)
        o = g
    pyr[n_lev - 1] = o
    return pyr

def nlp(img, n_lev, params):
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
        # Use register_buffer for parameters that are part of the model's state
        # but are not trainable. This ensures they are moved with .to(device).
        
        self.register_buffer('gamma', torch.tensor(2.60))
        
        filt0 = torch.tensor([[0.0400, 0.0400, 0.0500, 0.0400, 0.0400],
                              [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                              [0.0500, 0.0400, 0.0500, 0.0400, 0.0500],
                              [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                              [0.0400, 0.0400, 0.0500, 0.0400, 0.0400]],
                             dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.register_buffer('filt0', filt0)

        filt1 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0], [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]],
                             dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.register_buffer('filt1', filt1)
        
        sigmas = torch.tensor([0.1700, 4.8600], dtype=torch.float)
        self.register_buffer('sigmas', sigmas)

        F1_val = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                               [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                               [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                               [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                               [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                              dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.register_buffer('F1', F1_val)

        self.exp_s = 2.00
        self.exp_f = 0.60

    def forward(self, h_img, l_img, n_lev=None):
        # --- START OF FIX ---
        
        # The buffers are automatically on the correct device because the nn.Module (`self`)
        # was moved to the device in the main training script.
        
        if n_lev is None:
            # Calculate n_lev based on input shape. This is fine.
            n_lev = math.floor(math.log(min(h_img.shape[2:]), 2)) - 2

        # Create a temporary, local dictionary for the parameters for this forward pass.
        # This makes the forward pass stateless and XLA-friendly.
        # The tensors (self.F1, etc.) are already on the correct device.
        params = {
            'gamma': self.gamma,
            'F1': self.F1,
            'sigmas': self.sigmas,
            'filts': {
                0: self.filt0,
                1: self.filt1
            }
        }
        
        # Pass the local params dictionary to the nlp function.
        h_pyr = nlp(h_img, n_lev, params)
        l_pyr = nlp(l_img, n_lev, params)
        
        # --- END OF FIX ---
        
        dis = []
        for i in range(0, n_lev):
            diff = torch.pow(torch.abs(h_pyr[i] - l_pyr[i]), self.exp_s)
            diff_pow = torch.pow(torch.mean(torch.mean(diff, dim=-1), dim=-1), self.exp_f / self.exp_s)
            dis.append(diff_pow)

        dis = torch.cat(dis, -1)
        loss = torch.pow(torch.mean(dis, dim=-1), 1. / self.exp_f)

        return loss.mean()
