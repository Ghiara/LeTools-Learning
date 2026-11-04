"""DDPM scheduler extension exposing the log-probability API used by BPPO."""

from __future__ import annotations

import torch
from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from torch.distributions import Normal


class DDPMSchedulerWithLogProb(DDPMScheduler):
    """LeTools-compatible DDPM sampling plus per-element transition log-probs.

    RL-100's BPPO compares the probability of the same denoising transition
    under the frozen behavior policy and the trainable policy. Diffusers'
    standard DDPM scheduler samples that transition but does not return its
    probability, so this class exposes ``step_logprob`` and
    ``step_forward_logprob`` without changing the DDPM mean/variance formula.
    """

    def __init__(self, *args, logprob_min_std: float = 1e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.variance_type != "fixed_small":
            raise ValueError(
                "The RL-100 bridge currently supports DDPM variance_type='fixed_small' only; "
                f"got {self.config.variance_type!r}."
            )
        self.logprob_min_std = float(logprob_min_std)

    @staticmethod
    def _scalar_timestep(timestep: int | torch.Tensor) -> int:
        if isinstance(timestep, torch.Tensor):
            flat = timestep.detach().reshape(-1)
            if flat.numel() == 0:
                raise ValueError("Empty timestep tensor.")
            if flat.numel() > 1 and not torch.all(flat == flat[0]):
                raise ValueError("A DDPM batch must use one shared denoising timestep.")
            return int(flat[0].item())
        return int(timestep)

    def _previous_timestep(self, timestep: int) -> int:
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before sampling.")
        # Keep the log-probability path on exactly the same timestep schedule as
        # DDPMScheduler.step(), including non-default timestep spacing.
        previous_timestep = getattr(self, "previous_timestep", None)
        if callable(previous_timestep):
            previous = previous_timestep(timestep)
            if isinstance(previous, torch.Tensor):
                return int(previous.detach().reshape(-1)[0].item())
            return int(previous)
        # Compatibility with diffusers releases predating previous_timestep().
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        return timestep - step_ratio

    def _mean_std_and_x0(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        t = self._scalar_timestep(timestep)
        prev_t = self._previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t].to(
                device=sample.device, dtype=sample.dtype
            )
        else:
            # DDPMScheduler calls this tensor ``one``. ``final_alpha_cumprod``
            # belongs to DDIMScheduler and is not present on DDPM in current
            # diffusers releases.
            terminal_alpha = getattr(self, "one", None)
            if terminal_alpha is None:
                terminal_alpha = torch.tensor(1.0)
            alpha_prod_t_prev = terminal_alpha.to(device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"Unsupported prediction_type={self.config.prediction_type!r}")

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (
            alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        )
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        if t > 0:
            variance = self._get_variance(t).to(device=sample.device, dtype=sample.dtype)
            std = variance.clamp_min(self.logprob_min_std**2).sqrt()
        else:
            std = torch.as_tensor(
                self.logprob_min_std, device=sample.device, dtype=sample.dtype
            )
        return mean, std, pred_original_sample, t

    def step_logprob(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        eta: float = 1.0,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
        return_dict: bool = True,
        **_: object,
    ):
        del eta
        mean, std, pred_original_sample, t = self._mean_std_and_x0(
            model_output, timestep, sample
        )
        if t > 0:
            if variance_noise is None:
                variance_noise = torch.randn(
                    sample.shape,
                    generator=generator,
                    device=sample.device,
                    dtype=sample.dtype,
                )
            prev_sample = mean + std * variance_noise
        else:
            prev_sample = mean
        # DDPM's t=0 transition is deterministic. It has no density ratio and
        # must not create an artificial 1/std gradient in PPO.
        if t == 0:
            log_prob = torch.zeros_like(prev_sample)
        else:
            log_prob = Normal(mean, std).log_prob(prev_sample)
        if not return_dict:
            return (prev_sample,), log_prob
        return (
            DDPMSchedulerOutput(
                prev_sample=prev_sample, pred_original_sample=pred_original_sample
            ),
            log_prob,
        )

    def step_forward_logprob(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        next_sample: torch.Tensor,
        eta: float = 1.0,
        **_: object,
    ) -> torch.Tensor:
        del eta
        mean, std, _, t = self._mean_std_and_x0(model_output, timestep, sample)
        if t == 0:
            # Keep a zero-gradient connection so RL-100's unconditional
            # loss.backward() remains valid for this deterministic step.
            return torch.zeros_like(next_sample) + model_output * 0.0
        return Normal(mean, std).log_prob(next_sample)
