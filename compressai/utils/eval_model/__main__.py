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
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import os
import argparse
import json
import math
import sys
import time
import random

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torchvision

import compressai

from compressai.ops import compute_padding
from compressai.zoo import image_models as pretrained_models
from compressai.zoo.image import model_architectures as architectures

# from libtiff i
import cv2
import numpy as np

from .percentile import Percentile

from .functions import *
from natsort import os_sorted

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
HDR_EXTENSIONS = [
    '.hdr',
]


def is_hdr_file(filename):
    return any(filename.endswith(extension) for extension in HDR_EXTENSIONS)


def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for filename in os.listdir(rootpath):
        if is_hdr_file(filename):
            image_files.append(rootpath + '/{}'.format(filename))


    return os_sorted(image_files)


def read_image(filepath: str) -> torch.Tensor:
    # assert filepath.is_file()
    print("filepath: ", filepath)
    hdr = cv2.imread(filepath, flags=cv2.IMREAD_ANYDEPTH)
    # height = 512
    # scale_percent = 512 / hdr.shape[0]
    # width = int(hdr.shape[1] * scale_percent)
    # dim = (width, height)
    # hdr = cv2.resize(hdr, dim, interpolation=cv2.INTER_AREA)
    # hdr = hdr[0:512, 0:512]
    hdr = ((hdr - hdr.min()) / (hdr.max() - hdr.min()))
    img = torch.Tensor(hdr.transpose((2, 0, 1)))

    return img


@torch.no_grad()
def inference(model, x, count):
    x = x.unsqueeze(0)
    # print('max-x: ', torch.max(x))
    # print('min-x: ', torch.min(x))

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["hyper_z_shape"])
    dec_time = time.time() - start

    # out_dec["x_hat"] = F.pad(
    #     out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    # )

    target_hdr = x

    # hdr_hat = out_dec["x_hat"]
    # hdr_hat = ((hdr_hat - hdr_hat.min()) / (hdr_hat.max() - hdr_hat.min()))

    # hdr_min = 5.0
    # hdr_max = 1e7  # random.choice([1e3, 1e4, 1e5, 1e6, 1e7])

    # cali_hdr_hat = (hdr_max - hdr_min) * hdr_hat + hdr_min
    # cali_target_hdr = (hdr_max - hdr_min) * target_hdr + hdr_min

    ldr_hat = out_dec["ldr_x_hat"]
    ldr_hat = ((ldr_hat - ldr_hat.min()) / (ldr_hat.max() - ldr_hat.min()))

    ############################################
    ### Save LDR image
    ############################################
    # bgr2hsv = BGR_HSV()
    #
    # hsv_target_hdr = bgr2hsv(target_hdr)
    # hsv_ldr_hat = bgr2hsv(ldr_hat)

    # ldr_out_img = color_reproduce(ldr_hat, target_hdr, hsv_ldr_hat, hsv_target_hdr)
    # ldr_out_img = ldr_hat * 255
    # gamma = 2.2
    # ldr_out_img = torch.pow(ldr_out_img, 1 / gamma) * 255
    # print('max-ldr_out_img: ', torch.max(ldr_hat))
    # print('min-ldr_out_img: ', torch.min(ldr_hat))

    # ldr_out_img = ldr_out_img.squeeze()
    # # ldr_out_img = ldr_out_img[2, :, :] * 0.212656 + ldr_out_img[1, :, :] * 0.715158 + ldr_out_img[0, :, :] * 0.072186
    # ldr_out_img = ldr_out_img.numpy()
    # ldr_out_img = ldr_out_img.astype(np.uint8)
    # ldr_out_img = ldr_out_img.transpose((1, 2, 0))
    # cv2.imwrite(str(time.time()) + '_ldr_out_img.png', ldr_out_img)

    log_dir = "./test_ldr/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # re_hdr_save = np.squeeze(hdr_hat.cpu().numpy(), 0).transpose(1, 2, 0).astype(np.float32)
    # cv2.imwrite(log_dir + str(count) + '.hdr', re_hdr_save)

    sdr_save = np.squeeze((ldr_hat * 255).cpu().numpy(), 0).transpose(1, 2, 0).astype(np.uint8)
    cv2.imwrite(log_dir + 'sdr_layer_' + str(count) + '.png', sdr_save)

    # torchvision.utils.save_image(ldr_out_img, str(time.time()) + '_ldr_out_img.png')

    ############################################

    ############################################
    ### Save exposure sequence for gt and rec
    ############################################
    # wins = hdrMetric()
    # out_dir = 'ek_imgs/' + str(count) + '/'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # Qs, gt_seq, output_seq = wins(cali_hdr_hat, cali_target_hdr)
    # # print("Qs: ", len(Qs))
    # # print("gt_seq: ", len(gt_seq))
    # # print("out_seq: ", len(output_seq))
    # for e in range(len(Qs)):
    #
    #     gt_seq[e] = gt_seq[e].squeeze()
    #     gt_seq[e] = gt_seq[e].numpy() * 255
    #     gt_seq[e] = gt_seq[e].astype(np.uint8)
    #     gt_seq[e] = gt_seq[e].transpose((1, 2, 0))
    #     cv2.imwrite(out_dir + 'e_' + str(e) + 'gt_img.png', gt_seq[e])
    #
    #     output_seq[e] = output_seq[e].squeeze()
    #     output_seq[e] = output_seq[e].numpy() * 255
    #     output_seq[e] = output_seq[e].astype(np.uint8)
    #     output_seq[e] = output_seq[e].transpose((1, 2, 0))
    #
    #     cv2.imwrite(out_dir + 'e_' + str(e) + 'output_img.png', output_seq[e])
    #
    #     # torchvision.utils.save_image(gt_seq[e], out_dir + 'e_' + str(e) + 'gt_img.png')
    #     # torchvision.utils.save_image(output_seq[e], out_dir + 'e_' + str(e) + 'output_img.png')
    #     print("######################################")
    #     print("Q of " + str(e) + "-th exposure: ", Qs[e])
    #     print("######################################")
    ############################################

    ############################################
    ### Calculate average NLPD value for each channel
    ############################################
    ldr_metric = NLPD_Loss()
    # for channel in range(cali_ldr.shape[1]):
    nlpd_value = ldr_metric(target_hdr, ldr_hat)
    nlpd_value = nlpd_value.item()
    # NLPD_list.append(nlpd_value.item())

    # avg_NLPD = sum(NLPD_list) / len(NLPD_list)
    avg_NLPD = nlpd_value
    ############################################

    ############################################
    ### PU encoding for HDR image
    ############################################
    # out_encoding = PU21_encoding(cali_hdr_hat)
    # x_encoding = PU21_encoding(cali_target_hdr)
    #
    # print('max-x-encoding: ', torch.max(x_encoding))
    # print('min-x-encoding: ', torch.min(x_encoding))
    # ############################################
    #
    # max_value = x_encoding.max().item()
    # max_value = int(max_value)
    ############################################
    ### Compute distortion and bpp
    ############################################
    # metrics_hdr = compute_metrics(x_encoding, out_encoding, max_val=max_value)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    # bpp_hdr = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels  ## still using 8-bit coding?
    bpp_ldr = sum([len(out_enc["strings"][1][0]), len(out_enc["strings"][2][0])]) * 8.0 / num_pixels

    # print("out_enc[\"strings\"][0][0]: ", out_enc["strings"][0][0])
    # print("len(out_enc[\"strings\"][0][0]): ", len(out_enc["strings"][0][0]))
    # bpp_hdr = sum(
    #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
    #     for likelihoods in out_dec["likelihoods"].values()
    # ).item()

    return {
        # "psnr-PU21-hdr": metrics_hdr["psnr-PU21"],
        # "ms-ssim-PU21-hdr": metrics_hdr["ms-ssim-PU21"],
        # "bpp-hdr": bpp_hdr,
        "bpp-ldr": bpp_ldr,
        "NLPD-value": avg_NLPD,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True, progress=False
    ).eval()


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]

    model_cls = architectures[arch]
    net = model_cls.from_state_dict(state_dict)
    if not no_update:
        net.update(force=True)
    return net.eval()


def eval_model(
    model: nn.Module,
    outputdir: Path,
    inputdir: Path,
    filepaths,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    count = 0
    for filepath in filepaths:
        count += 1
        x = read_image(filepath).to(device)

        print(f'Count={count}.')
        if not entropy_estimation:
            if args["half"]:
                model = model.half()
                x = x.half()
            rv = inference(model, x, count)
        else:
            print("Error: entropy_estimation = True")
            # rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v
        if args["per_image"]:
            if not Path(outputdir).is_dir():
                raise FileNotFoundError("Please specify output directory")

            output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(
                inputdir
            )
            output_subdir.mkdir(parents=True, exist_ok=True)
            image_metrics_path = output_subdir / f"{filepath.stem}-{trained_net}.json"
            with image_metrics_path.open("wb") as f:
                output = {
                    "source": filepath.stem,
                    "name": args["architecture"],
                    "description": f"Inference ({description})",
                    "results": rv,
                }
                f.write(json.dumps(output, indent=2).encode())

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    # Common options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parent_parser.add_argument(
        "-d",
        "--output_directory",
        type=str,
        default="",
        help="path of output directory. Optional, required for output json file, results per image. Default will just print the output results.",
    )
    parent_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )
    parent_parser.add_argument(
        "--per-image",
        action="store_true",
        help="store results for each image of the dataset, separately",
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        type=str,
        default="1",
        help="Pretrained model qualities. (example: '1,2,3,4') (default: %(default)s)",
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="checkpoint_paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    if args.source not in ["checkpoint", "pretrained"]:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    # create output directory
    if args.output_directory:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    if args.source == "pretrained":
        args.qualities = [int(q) for q in args.qualities.split(",") if q]
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    else:
        runs = args.checkpoint_paths
        opts = (args.architecture, args.no_update)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, run)
        if args.source == "pretrained":
            trained_net = f"{args.architecture}-{args.metric}-{run}-{description}"
        else:
            cpt_name = Path(run).name[: -len(".tar.pth")]  # removesuffix() python3.9
            trained_net = f"{cpt_name}-{description}"
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        args_dict = vars(args)
        metrics = eval_model(
            model,
            args.output_directory,
            args.dataset,
            filepaths,
            trained_net=trained_net,
            description=description,
            **args_dict,
        )
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }
    if args.output_directory:
        output_file = (
            args.output_file
            if args.output_file
            else f"{args.architecture}-{description}"
        )

        with (Path(f"{args.output_directory}/{output_file}").with_suffix(".json")).open(
            "wb"
        ) as f:
            f.write(json.dumps(output, indent=2).encode())

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    now = time.time()
    main(sys.argv[1:])
    print(time.time() - now)
