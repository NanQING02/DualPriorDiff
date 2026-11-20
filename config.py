"""Command line options shared by train.py and sample.py."""

import argparse
import os

import torch

# Hyper-parameters reported in the paper.
PAPER_DEFAULTS = dict(
    num_epoch=400,
    batch_size=4,
    learning_rate=5e-6,          # diffusion network
    priori_learning_rate=6e-8,   # prior acquisition network
    beta1=0.9,
    beta2=0.99,
    image_size=256,
)

# Encoder layout of the reported model: four PIAR levels followed by two
# self-attention levels.
DEFAULT_ENCODER = "PIAR,PIAR,PIAR,PIAR,RAB,RAB"


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "yes", "y", "on")


def build_parser():
    parser = argparse.ArgumentParser(description="DualPriorDiff")

    # ---- data ----
    parser.add_argument("--train_gt_dir", type=str, default="",
                        help="folder of ground-truth training images (256x256)")
    parser.add_argument("--train_masked_dir", type=str, default="",
                        help="folder of masked training images, same file names as --train_gt_dir")
    parser.add_argument("--val_masked_dir", type=str, default="",
                        help="folder of masked validation images")
    parser.add_argument("--val_gt_dir", type=str, default="",
                        help="folder of ground-truth validation images (optional, only for metrics)")

    # ---- inference ----
    parser.add_argument("--masked_dir", type=str, default="",
                        help="folder of masked images to inpaint (sample.py)")
    parser.add_argument("--weights", type=str, default="",
                        help="denoising network checkpoint file (sample.py)")
    parser.add_argument("--priori_weights", type=str, default="",
                        help="prior network checkpoint file (sample.py)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="folder for the inpainted images (sample.py)")

    # ---- output ----
    parser.add_argument("--out_dir", type=str, default="training_results",
                        help="folder for validation images")
    parser.add_argument("--weights_dir", type=str, default="weights",
                        help="folder for denoising network checkpoints")
    parser.add_argument("--priori_weights_dir", type=str, default="priori_weights",
                        help="folder for prior network checkpoints")

    # ---- optimisation (values as reported in the paper) ----
    parser.add_argument("--num_epoch", type=int, default=PAPER_DEFAULTS["num_epoch"])
    parser.add_argument("--epoch_restore", type=int, default=0,
                        help="resume from this epoch; 0 starts from scratch")
    parser.add_argument("--batch_size", type=int, default=PAPER_DEFAULTS["batch_size"])
    parser.add_argument("--learning_rate", type=float, default=PAPER_DEFAULTS["learning_rate"])
    parser.add_argument("--priori_learning_rate", type=float, default=PAPER_DEFAULTS["priori_learning_rate"])
    parser.add_argument("--beta1", type=float, default=PAPER_DEFAULTS["beta1"])
    parser.add_argument("--beta2", type=float, default=PAPER_DEFAULTS["beta2"])
    parser.add_argument("--image_size", type=int, default=PAPER_DEFAULTS["image_size"])
    parser.add_argument("--mask_percent", type=float, default=0.5,
                        help="fraction of the fixed square region that is dropped by the on-the-fly mask")
    parser.add_argument("--valid_num", type=int, default=20,
                        help="number of validation images written every 10 epochs; -1 for all")

    # ---- ablation switches ----
    parser.add_argument("--prior", type=str, default="on", choices=["on", "off"],
                        help="on: PIAR blocks inject the prior vector. off: no prior injection (baseline).")
    parser.add_argument("--skip", type=str, default="dense", choices=["dense", "plain"],
                        help="dense: every encoder layer output is routed to the decoder. "
                             "plain: standard U-Net skip.")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                        help="per-level encoder block types, comma separated, one entry per level. "
                             "PIAR / RAB / RB. Example: 2PIAR-1RB-2RAB -> PIAR,PIAR,RB,RAB,RAB")

    # ---- runtime ----
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser


def parse_encoder(spec):
    """Accept both 'PIAR,PIAR,RB,RAB,RAB' and '2PIAR-1RB-2RAB'."""
    spec = spec.strip()
    if "-" in spec:
        blocks = []
        for part in spec.split("-"):
            part = part.strip().upper()
            i = 0
            while i < len(part) and part[i].isdigit():
                i += 1
            count, name = (int(part[:i]) if i else 1), part[i:]
            if name not in ("PIAR", "RAB", "RB"):
                raise ValueError("unknown block type %r in encoder spec %r" % (name, spec))
            blocks.extend([name] * count)
        return ",".join(blocks)
    return spec


def prepare(argv=None):
    args = build_parser().parse_args(argv)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "19198")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    world_size = max(torch.cuda.device_count(), 1)
    os.environ["WORLD_SIZE"] = str(world_size)

    args.encoder = parse_encoder(args.encoder)
    args.world_size = world_size
    args.use_prior = (args.prior == "on")
    return args
