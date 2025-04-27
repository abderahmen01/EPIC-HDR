import argparse
import random
import shutil
import sys
import math
import os
import os.path
import skimage.color as color
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import imageio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from compressai.datasets.image import ImageFolder
from compressai.losses.NLPD import NLPD_Loss
from compressai.losses.rate_distortion import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from tqdm import tqdm
tonemap = lambda x: (np.log(np.clip(x, 0, 1) * 5000 + 1) / np.log(5000 + 1) * 255).astype(np.uint8)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        with torch.autograd.set_detect_anomaly(True):
            x = d['nlp_I']
            hdr_name = d['hdr_name']
            hdr_l = d['hdr_l']
            s_max = d['s_max']
            hdr = d['hdr']
            reference = d['reference']
            hdr_l = hdr_l.to(device)
            s_max = s_max.to(device).to(torch.float32)
            x = x.to(device)
            hdr = hdr.to(device)
            reference = reference.to(device)

            optimizer.zero_grad()
            out_net1, out_net2 = model(x, s_max, hdr)
            out_criterion = criterion(out_net1, out_net2, hdr_l, reference, s_max)
            out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 2 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)/4}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tLDR loss: {out_criterion["ldr_loss"].item():.6f} |'
                f'\tBpp loss1: {out_criterion["bpp_loss1"].item():.2f} |'
                f'\tHDR loss: {out_criterion["hdr_loss"].item():.6f} |'
                f'\tBpp loss2: {out_criterion["bpp_loss2"].item():.2f} |'
            )


def test_epoch(epoch, test_dataloader, criterion, path, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    ldr_loss = AverageMeter()
    bpp_loss1 = AverageMeter()
    bpp_loss2 = AverageMeter()
    hdr_loss = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            x = d['nlp_I']
            hdr_name = d['hdr_name']
            hdr_l = d['hdr_l']
            s_max = d['s_max']
            hdr = d['hdr']
            hdr_l = hdr_l.to(device)
            s_max = s_max.to(device).to(torch.float32)
            x = x.to(device)
            hdr = hdr.to(device)

            ###
            out_net1, out_net2 = model(x, s_max, hdr)
            ldr_hat = out_net1["ldr_x_hat"]

            ldr_hat_v = ldr_hat[:, 2, :, :].unsqueeze(1)
            ldr_hat_hs = ldr_hat[:, 0:2, :, :]

            ldr_out_v2 = (300 - 5) * ldr_hat_v + 5
            ldr_out2 = torch.cat([ldr_hat_hs, ldr_out_v2], dim=1)

            _save_image(ldr_out2, os.path.join(path, 'ldr'), hdr_name[0])

            reference = d['reference']
            hdr_name = d['hdr_name']
            reference = reference.to(device)
            out_criterion = criterion(out_net1, out_net2, hdr_l, reference, s_max)

            hdr_recon = out_net2["hdr"]
            hdr_recon_max = torch.max(torch.max(torch.max(hdr_recon, 1)[0], 1)[0], 1)[0].unsqueeze(1).unsqueeze(
                1).unsqueeze(1)
            hdr_recon = hdr_recon / (hdr_recon_max + 1e-30)
            hdr_rgb = hdr_recon.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb8_h_tm = tonemap(hdr_rgb / np.max(hdr_rgb))
            filename1 = os.path.join(path, 'hdr/' + hdr_name[0] + 'h_tm.png')
            filename2 = os.path.join(path, 'hdr/' + hdr_name[0] + '_hdr.exr')
            imageio.imwrite(filename1, rgb8_h_tm)
            imageio.imwrite(filename2, hdr_rgb)

            loss.update(out_criterion["loss"])
            bpp_loss1.update(out_criterion["bpp_loss1"])
            ldr_loss.update(out_criterion["ldr_loss"])
            bpp_loss2.update(out_criterion["bpp_loss2"])
            hdr_loss.update(out_criterion["hdr_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tLDR loss: {ldr_loss.avg:.6f} |"
        f"\tBpp loss1: {bpp_loss1.avg:.2f} |"
        f"\tHDR loss: {hdr_loss.avg:.6f} |"
        f"\tBpp loss2: {bpp_loss2.avg:.2f} |"
    )

    return loss.avg


def _save_image(img, path, name):
    # color
    d_max = 300
    d_min = 5
    hdr_h = img.data[0].permute(1, 2, 0)
    t = hdr_h[:, :, 2].cpu()
    t[t > d_max] = d_max
    t[t < d_min] = d_min
    t = (t - d_min) / (d_max - d_min)
    t = (t ** (1 / 2.2))
    hdr_h = hdr_h.cpu().numpy()
    hdr_h[:, :, 2] = t.squeeze().cpu().numpy()
    hdr_h[:, :, 1] = hdr_h[:, :, 1] * 0.6
    result = color.hsv2rgb(hdr_h)
    sz = result.shape
    result1 = np.zeros((sz))
    result1[:, :, 0] = (result[:, :, 0] - np.min(result[:, :, 0])) / (np.max(result[:, :, 0]) - np.min(result[:, :, 0]))
    result1[:, :, 1] = (result[:, :, 1] - np.min(result[:, :, 1])) / (np.max(result[:, :, 1]) - np.min(result[:, :, 1]))
    result1[:, :, 2] = (result[:, :, 2] - np.min(result[:, :, 2])) / (np.max(result[:, :, 2]) - np.min(result[:, :, 2]))
    plt.imsave(path + name[:-4] + '.png', result1)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        torch.save(state["state_dict"], "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default='./train', help="Training dataset"
    )
    parser.add_argument(
        "--test_dataset", type=str, default='./test', help="Test dataset"
    )
    parser.add_argument("--results_savepath", type=str, default='./output')
    parser.add_argument(
        "-e",
        "--epochs",
        default=600,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=200,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )


    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument("--epochs_per_eval", type=int, default=500)
    parser.add_argument("--epochs_per_save", type=int, default=100)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument(
        "--save", type=bool, default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, default='', help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = ImageFolder(os.path.join(args.dataset, 'train.txt'), args.dataset, train=True, test=False)
    test_dataset = ImageFolder(os.path.join(args.test_dataset, 'test.txt'), args.test_dataset, train=False, test=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net1 = image_models["mbt2018"](quality=3)
    net1 = net1.to(device)

    net2 = image_models["ldr2hdr"](quality=3)
    net2 = net2.to(device)

    net = image_models["end2end"](net1, net2)
    net = net.to(device)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    if args.checkpoint != '':
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    else:
        last_epoch = 0

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):

        train_one_epoch(
           net,
           criterion,
           train_dataloader,
           optimizer,
           epoch,
           args.clip_max_norm,
        )
        lr_scheduler.step()
        if (epoch+1) % 50 == 0 or epoch == 0:
            if not os.path.exists(args.results_savepath):
                os.makedirs(os.path.join(args.results_savepath, 'hdr'))
                os.makedirs(os.path.join(args.results_savepath, 'ldr'))
            loss = test_epoch(epoch, test_dataloader, criterion, args.results_savepath, net)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            save_checkpoint(
               {
                   "epoch": epoch,
                   "state_dict": net.state_dict(),
                   "loss": loss,
                   "optimizer": optimizer.state_dict(),
                   "lr_scheduler": lr_scheduler.state_dict(),
               },
               is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
