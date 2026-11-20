"""Quantitative evaluation of a result folder against the ground truth.

    python evaluate.py --gt_dir data/celeba_hq/test/gt --pred_dir results/celeba_hq/center

Reports PSNR, SSIM and LPIPS, all averaged over the image pairs found in both
folders. Files are matched by name.
"""

import argparse
import os

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def load_rgb(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="PSNR / SSIM / LPIPS")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    names = sorted(n for n in os.listdir(args.gt_dir) if n.lower().endswith(IMG_EXT))
    pred_names = set(os.listdir(args.pred_dir))

    lpips_fn = lpips.LPIPS(net='alex').to(args.device).eval()
    psnr_list, ssim_list, lpips_list = [], [], []

    for name in names:
        if name not in pred_names:
            print("missing prediction, skipped:", name)
            continue
        gt = load_rgb(os.path.join(args.gt_dir, name))
        pred = load_rgb(os.path.join(args.pred_dir, name))

        psnr_list.append(psnr_metric(gt, pred, data_range=1.0))
        ssim_list.append(ssim_metric(gt, pred, data_range=1.0, channel_axis=2))

        gt_t = torch.from_numpy(gt).permute(2, 0, 1)[None] * 2 - 1
        pred_t = torch.from_numpy(pred).permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad():
            lpips_list.append(lpips_fn(gt_t.to(args.device), pred_t.to(args.device)).item())

    if not psnr_list:
        raise SystemExit("no image pairs found")

    print("images : %d" % len(psnr_list))
    print("PSNR   : %.4f" % float(np.mean(psnr_list)))
    print("SSIM   : %.4f" % float(np.mean(ssim_list)))
    print("LPIPS  : %.4f" % float(np.mean(lpips_list)))


if __name__ == "__main__":
    main()
