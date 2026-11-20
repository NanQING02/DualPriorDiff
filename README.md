# DualPriorDiff

Official implementation of **DualPriorDiff: Dual-Conditioned Prior-Guided Diffusion
Network for Large-Scale Semantic-consistency Image Inpainting**.

The network combines three components:

* a lightweight **Prior Information Acquisition Network (PIAN)** that compresses the
  visible content of a masked image into a 256-dimensional prior vector,
* a **Prior-Infused Attention Residual Block (PIAR)** that injects the prior vector
  into every denoising step, and
* **dense skip connections** between the encoder and the decoder.

The masked image and the prior vector form a dual condition that guides the reverse
diffusion process. Sampling uses deterministic DDIM.

## Repository layout

```
config.py                 shared command line options (training and inference)
train.py                  alternating joint training of the two networks
sample.py                 inference with a trained checkpoint
evaluate.py               PSNR / SSIM / LPIPS between two folders
diffusion_scheduler.py    Gaussian diffusion (DDPM) with DDIM sampling
metrics.py                image conversion and saving helpers
models/
    unet.py               denoising U-Net, PIAR blocks, dense skip connections
    prior_net.py          PIAN
data/
    dataset.py            masked / ground-truth datasets
    make_masks.py         generator of the five mask types
```

## Requirements

Python 3.8+ and a CUDA-capable GPU.

```bash
pip install -r requirements.txt
```

## Data preparation

Images are 256x256 RGB. Ground-truth and masked images are stored in separate
folders that use the same file names; pairs are matched in sorted file-name order.

Generate the masked counterparts of an image folder with `data/make_masks.py`. The
five mask types used in the paper:

```bash
# center mask: fixed black square in the middle
python -m data.make_masks --type center      --input $GT_DIR --output $CENTER_DIR --square_size 120

# random path mask: random thick strokes
python -m data.make_masks --type random_path --input $GT_DIR --output $PATH_DIR

# striped mask: horizontal stripes
python -m data.make_masks --type striped     --input $GT_DIR --output $STRIPED_DIR

# 2x super-resolution mask: checkerboard
python -m data.make_masks --type sr          --input $GT_DIR --output $SR_DIR

# half mask: a block covering 40%-60% of one side
python -m data.make_masks --type half        --input $GT_DIR --output $HALF_DIR
```

Run `python -m data.make_masks --help` for the parameters of each mode
(`--square_size`, `--min_brush_size` / `--max_brush_size` / `--route_length`,
`--stripe_width` / `--stripe_distance`, `--mask_color`, `--min_ratio` / `--max_ratio`).

## Training

```bash
python train.py \
    --train_gt_dir      /path/to/train/gt \
    --train_masked_dir  /path/to/train/center \
    --val_masked_dir    /path/to/val/center \
    --out_dir           training_results \
    --weights_dir       weights \
    --priori_weights_dir priori_weights
```

Hyper-parameters are the ones reported in the paper and are already the defaults:

| Option | Value |
| --- | --- |
| `--num_epoch` | 400 |
| `--batch_size` | 4 |
| `--learning_rate` | 5e-6 (diffusion network) |
| `--priori_learning_rate` | 6e-8 (prior network) |
| `--beta1`, `--beta2` | 0.9, 0.99 |
| `--image_size` | 256 |
| `--mask_percent` | 0.5 |

The two networks are optimised jointly: the prior network minimises the L1 distance
between the prior vectors of the masked image and of the ground truth, while the
denoising network minimises the diffusion loss with the prior vector detached.

Multiple GPUs are selected with `--gpu`, which sets `CUDA_VISIBLE_DEVICES`:

```bash
python train.py --gpu 0,1,2,3 --train_gt_dir ... --train_masked_dir ... --val_masked_dir ...
```

Validation images and checkpoints are written every 10 epochs. Training resumes with
`--epoch_restore <epoch>`.

## Inference

```bash
python sample.py \
    --masked_dir     /path/to/test/center \
    --weights        weights/epoch180 \
    --priori_weights priori_weights/epoch180 \
    --output_dir     results/celeba_hq/center
```

## Evaluation

```bash
python evaluate.py --gt_dir /path/to/test/gt --pred_dir results/celeba_hq/center
```

Prints the average PSNR, SSIM and LPIPS over the image pairs found in both folders.

## Checkpoints

Trained weights for CelebA-HQ (256x256) are available from Zenodo:

> **https://doi.org/10.5281/zenodo.XXXXXXX**  <!-- replace after uploading to Zenodo -->

Each release contains the denoising network and the prior network of the reported
model. Download both files and pass them to `sample.py` through `--weights` and
`--priori_weights`.

## Ablation configurations

The ablation tables of the paper can be reproduced with three options.

`--encoder` selects the block type of every encoder level (one entry per level,
separated by commas):

| `--encoder` | Level layout |
| --- | --- |
| `PIAR,PIAR,PIAR,PIAR,RAB,RAB` | default, reported model |
| `PIAR,PIAR,RB,RAB,RAB,RAB` | example, mixed layout |
| `RB,RB,RB,RB,RB,RB` | all plain residual blocks |

`PIAR` is a prior-infused attention residual block, `RAB` a residual block with
self-attention and `RB` a plain residual block. The shorthand form used in the paper
is accepted as well, for example `--encoder 2PIAR-1RB-2RAB`.

`--prior` turns prior injection on or off:

| `--prior` | Effect |
| --- | --- |
| `on` | `PIAR` levels inject the prior vector (default) |
| `off` | `PIAR` levels fall back to plain residual blocks, the prior vector is ignored |

`--skip` selects the skip routing between encoder and decoder:

| `--skip` | Effect |
| --- | --- |
| `dense` | every encoder layer output is routed to the mirror decoder block (default) |
| `plain` | only the first decoder block of each resolution receives the encoder stage output |

Combining the three options covers the no-prior, no-dense-skip and combined
configurations of the ablation study, for example:

```bash
# without prior injection
python train.py --prior off    --train_gt_dir ... --train_masked_dir ... --val_masked_dir ...

# standard U-Net skip instead of dense skip
python train.py --skip plain   --train_gt_dir ... --train_masked_dir ... --val_masked_dir ...

# full model
python train.py --prior on --skip dense --encoder PIAR,PIAR,PIAR,PIAR,RAB,RAB \
                --train_gt_dir ... --train_masked_dir ... --val_masked_dir ...
```

## Notes

* The prior dimension is fixed at 256, which matches `models/unet.py`.
* Checkpoints saved by this code can be loaded directly; a `module.` prefix produced
  by other training wrappers is stripped automatically by `sample.py`.

## Contact

Nanqing Cao — nanqing.cao@hhu.edu.cn

College of Artificial Intelligence and Automation, Hohai University
