"""Train DualPriorDiff.

Single GPU:
    python train.py --train_gt_dir data/celeba_hq/train/gt \
                    --train_masked_dir data/celeba_hq/train/center \
                    --val_masked_dir data/celeba_hq/val/center

Multi GPU (torchrun-free, uses CUDA_VISIBLE_DEVICES):
    python train.py --gpu 0,1,2,3 --train_gt_dir ... --train_masked_dir ... --val_masked_dir ...

The two networks are optimised jointly: the prior network is updated by the L1
loss between the prior vectors of the masked image and of the ground truth, and
the denoising network is updated by the diffusion loss with the prior vector
detached.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import metrics
from config import prepare
from data.dataset import PairedMaskedDataset, MaskedDataset
from diffusion_scheduler import GaussianDiffusion
from models.prior_net import Prior_Extraction_Network
from models.unet import create_unet


def set_device(x):
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None:
                x[key] = item.cuda()
    elif isinstance(x, list):
        x = [item.cuda() for item in x]
    else:
        x = x.cuda()
    return x


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def build_models(args):
    model = create_unet(encoder_blocks=args.encoder,
                        use_prior=args.use_prior,
                        skip_mode=args.skip,
                        image_size=args.image_size).cuda()
    prior_net = Prior_Extraction_Network().cuda()

    if args.epoch_restore > 0:
        prior_net.load_state_dict(torch.load(
            os.path.join(args.priori_weights_dir, "epoch%d" % args.epoch_restore), map_location="cuda"))
        model.load_state_dict(torch.load(
            os.path.join(args.weights_dir, "epoch%d" % args.epoch_restore), map_location="cuda"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 betas=(args.beta1, args.beta2))
    optimizer_priori = torch.optim.Adam(prior_net.parameters(), lr=args.priori_learning_rate,
                                        betas=(args.beta1, args.beta2))
    return model, prior_net, optimizer, optimizer_priori


def main(local_rank, args):
    init_ddp(local_rank)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_dataset = PairedMaskedDataset(args.train_gt_dir, args.train_masked_dir, split='train')
    valid_dataset = MaskedDataset(args.val_masked_dir, limit=-1)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    g = get_ddp_generator(args.seed)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               drop_last=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               generator=g)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1)

    model, prior_net, optimizer, optimizer_priori = build_models(args)
    model = DDP(model, device_ids=[local_rank])
    scheduler = GaussianDiffusion(denoise_fn=model, image_size=args.image_size)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(max(args.epoch_restore, 1), args.num_epoch + 1):
        model.train()
        prior_net.train()
        running, running_prior, idx = 0.0, 0.0, 1
        optimizer.zero_grad()
        optimizer_priori.zero_grad()

        if is_main_process():
            train_loader = tqdm(train_loader, file=sys.stdout)

        for data, _ in train_loader:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                set_device(data)
                k_value = prior_net(data['MOS'])
                gd_value = prior_net(data['HR'])
                loss = scheduler.p_losses(data, k_value.detach())
                priori_loss = prior_net.priori_losses(gd_value, k_value)

            running += loss.item()
            running_prior += priori_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scaler.scale(priori_loss).backward()
            scaler.step(optimizer_priori)
            scaler.update()

            optimizer.zero_grad()
            optimizer_priori.zero_grad()

            if is_main_process():
                train_loader.desc = "[epoch %d/%d] loss: %.4f  priori_loss: %.9f" % (
                    epoch, args.num_epoch, running / idx, running_prior / idx)
            idx += 1

        dist.barrier()

        if epoch % 10 == 0 and is_main_process():
            save_samples(args, epoch, model, prior_net, scheduler, valid_loader)
            torch.save(model.module.state_dict(), os.path.join(args.weights_dir, "epoch%d" % epoch))
            torch.save(prior_net.state_dict(), os.path.join(args.priori_weights_dir, "epoch%d" % epoch))

    dist.destroy_process_group()


def save_samples(args, epoch, model, prior_net, scheduler, valid_loader):
    epoch_dir = os.path.join(args.out_dir, "epoch%d" % epoch)
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

    model.eval()
    prior_net.eval()
    with torch.no_grad():
        for i, (masked, _) in enumerate(valid_loader):
            if args.valid_num != -1 and i >= args.valid_num:
                break
            masked = masked.cuda()
            k_value = prior_net(masked)
            metrics.save_img(metrics.tensor2img(masked.detach().float().cpu()),
                             os.path.join(epoch_dir, "epoch%d_%d_masked.jpg" % (epoch, i)))
            rebuilt = scheduler.ddim_super_resolution(masked, k_value)
            metrics.save_img(metrics.tensor2img(rebuilt.detach().float().cpu()),
                             os.path.join(epoch_dir, "epoch%d_%d_ddim.jpg" % (epoch, i)))
    model.train()
    prior_net.train()


def make_dirs(args):
    for folder in (args.out_dir, args.weights_dir, args.priori_weights_dir):
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == "__main__":
    args = prepare()
    make_dirs(args)
    if args.world_size > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size)
    else:
        main(0, args)
