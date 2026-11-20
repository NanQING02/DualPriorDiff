import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device = next(self.denoise_fn.parameters()).device
        self.loss_type = loss_type
        self.conditional = conditional

        self.set_loss(self.device)
        self.set_new_noise_schedule(self.device)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule="linear",
            n_timestep=1000,
            linear_start=1e-6,
            linear_end=2e-2)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in):
        device = self.betas.device
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)

        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)

            return img

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in):
        return self.p_sample_loop(x_in)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, k_value, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['MOS'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod, k_value)

        loss = self.loss_func(noise, x_recon)
        return loss

    @torch.no_grad()
    def ddim_p_mean_variance(self, x, t, k_value, clip_denoised=True, condition_x=None):
        batch_size = x.shape[0]
        if t == 995:
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t + 5]]).repeat(batch_size, 1).to(x.device)
        else:
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t + 6]]).repeat(batch_size, 1).to(x.device)

        eps = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level, k_value)

        if t == 995:
            x_start = self.predict_start_from_noise(x, t=t + 4, noise=eps)
        else:
            x_start = self.predict_start_from_noise(x, t=t + 5, noise=eps)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        sigma = torch.Tensor([0]).cuda()
        # sigma = 0.5 * sigma
        noise = torch.randn_like(x)

        model_mean = self.sqrt_alphas_cumprod[t] * x_start + \
                     eps * torch.sqrt(1 - self.alphas_cumprod[t] - sigma ** 2) + \
                     sigma * noise
        return model_mean

    @torch.no_grad()
    def ddim_p_sample(self, x, t, k_value, clip_denoised=True, condition_x=None):
        model_mean = self.ddim_p_mean_variance(
            x=x, t=t, k_value=k_value, clip_denoised=clip_denoised, condition_x=condition_x)
        return model_mean

    @torch.no_grad()
    def ddim_p_sample_loop(self, x_in, k_value):
        device = self.betas.device
        use_timesteps = space_timesteps(len(self.betas), "ddim200")
        x = x_in
        shape = x.shape
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if i in use_timesteps:
                img = self.ddim_p_sample(img, i, k_value, condition_x=x)
        return img

    @torch.no_grad()
    def ddim_super_resolution(self, x_in, k_value):
        return self.ddim_p_sample_loop(x_in, k_value)


class DDIM_GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        conditional=True,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device = next(self.denoise_fn.parameters()).device
        self.conditional = conditional

        self.set_new_noise_schedule(self.device)

    def set_new_noise_schedule(self, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule="linear",
            n_timestep=1000,
            linear_start=1e-6,
            linear_end=2e-2)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * (x_t - noise * self.sqrt_one_minus_alphas_cumprod[t])


    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None):
        batch_size = x.shape[0]
        if t == 1990:
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t + 10]]).repeat(batch_size, 1).to(x.device)
        else:
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t + 11]]).repeat(batch_size, 1).to(x.device)

        eps = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)

        if t == 1990:
            x_start = self.predict_start_from_noise(x, t=t + 9, noise=eps)
        else:
            x_start = self.predict_start_from_noise(x, t=t + 10, noise=eps)

        if clip_denoised:
            x_start.clamp_(-1., 1.)
        # if t == 1990:
        #     sigma = (torch.sqrt(1 - self.alphas_cumprod[t]) / self.sqrt_one_minus_alphas_cumprod[t+19]) * \
        #             (self.sqrt_one_minus_alphas_cumprod[t+19] / torch.sqrt(self.alphas_cumprod[t]))
        # else:
        #     sigma = (torch.sqrt(1 - self.alphas_cumprod[t]) / self.sqrt_one_minus_alphas_cumprod[t+20]) * \
        #             (self.sqrt_one_minus_alphas_cumprod[t+20] / torch.sqrt(self.alphas_cumprod[t]))
        sigma = torch.Tensor([0]).to("cuda:0")
        # sigma = 0.5 * sigma
        noise = torch.randn_like(x)

        model_mean = self.sqrt_alphas_cumprod[t] * x_start + \
                     eps * torch.sqrt(1 - self.alphas_cumprod[t] - sigma ** 2) + \
                     sigma * noise
        return model_mean

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        return model_mean

    @torch.no_grad()
    def ddim_p_sample_loop(self, x_in):
        device = self.betas.device
        use_timesteps = space_timesteps(len(self.betas), "ddim200")
        x = x_in
        shape = x.shape
        img = torch.randn(shape).to(device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if i in use_timesteps:
                img = self.p_sample(img, i, condition_x=x)
        return img

    # def set_new_steptimes(self, use_timesteps):
        # to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        # sqrt_alphas_cumprod_prev = []
        # alphas_cumprod_prev = []
        # sqrt_one_minus_alphas_cumprod = []
        #
        # for i in reversed(range(0, self.num_timesteps)):
        #     if i in use_timesteps:
        #
        # self.num_timesteps = len(use_timesteps)


    @torch.no_grad()
    def ddim_super_resolution(self, x_in):
        return self.ddim_p_sample_loop(x_in)


if __name__ == "__main__":
    model = create_sr256_unet().cuda()
    scheduler = GaussianDiffusion(denoise_fn=model, image_size=256)
    HR = torch.Tensor(2, 3, 256, 256).to("cuda:0")
    LR = torch.Tensor(2, 3, 256, 256).to("cuda:0")
    data = {'HR': HR, 'MOS': LR}
    loss = scheduler.p_losses(data)
