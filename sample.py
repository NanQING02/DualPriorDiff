"""Run inference with a trained checkpoint.

    python sample.py --masked_dir data/celeba_hq/test/center \
                     --weights weights/epoch180 \
                     --priori_weights priori_weights/epoch180 \
                     --output_dir results/celeba_hq/center \
                     --batch_size 8

The masked image is compressed into a prior vector by the prior network and the
image is reconstructed by deterministic DDIM sampling.
"""

import os

import torch
from tqdm import tqdm

import metrics
from config import prepare
from data.dataset import MaskedInferenceDataset
from diffusion_scheduler import GaussianDiffusion
from models.prior_net import Prior_Extraction_Network
from models.unet import create_unet


def load_state(path, model):
    state = torch.load(path, map_location="cpu")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def main():
    args = prepare()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    dataset = MaskedInferenceDataset(args.masked_dir)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers)

    model = create_unet(encoder_blocks=args.encoder,
                        use_prior=args.use_prior,
                        skip_mode=args.skip,
                        image_size=args.image_size)
    model = load_state(args.weights, model).cuda().eval()

    prior_net = Prior_Extraction_Network()
    prior_net = load_state(args.priori_weights, prior_net).cuda().eval()

    scheduler = GaussianDiffusion(denoise_fn=model, image_size=args.image_size)

    with torch.no_grad():
        for masked, names in tqdm(loader):
            masked = masked.cuda()
            k_value = prior_net(masked)
            rebuilt = scheduler.ddim_super_resolution(masked, k_value)
            rebuilt = metrics.tensor2img(rebuilt.detach().float().cpu())
            metrics.save_img(rebuilt, os.path.join(args.output_dir, names[0]))

    print("results written to", args.output_dir)


if __name__ == "__main__":
    main()
