"""Expose a trained LeTools Diffusion Policy through RL-100's actor API."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from .normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

from .scheduler_logprob import DDPMSchedulerWithLogProb


def load_diffusion_config_rl100(checkpoint_dir: str | Path) -> DiffusionConfig:
    """Load a registered LeRobot policy config and require Diffusion Policy."""
    config = PreTrainedConfig.from_pretrained(
        str(Path(checkpoint_dir).expanduser().resolve()),
        local_files_only=True,
    )
    if not isinstance(config, DiffusionConfig):
        raise TypeError(
            "RL-100 bridge requires a LeRobot DiffusionPolicy checkpoint, "
            f"but loaded {type(config).__name__}."
        )
    return config


def _minmax_field(stats: dict[str, torch.Tensor], key: str, eps: float = 1e-8):
    minimum = stats[f"{key}.min"].to(torch.float32)
    maximum = stats[f"{key}.max"].to(torch.float32)
    mean = stats[f"{key}.mean"].to(torch.float32)
    std = stats[f"{key}.std"].to(torch.float32)
    denominator = maximum - minimum
    denominator = torch.where(denominator == 0, torch.full_like(denominator, eps), denominator)
    scale = 2.0 / denominator
    offset = -1.0 - scale * minimum
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict={"min": minimum, "max": maximum, "mean": mean, "std": std},
    )


def load_checkpoint_normalizer(checkpoint_dir: str | Path) -> tuple[LinearNormalizer, dict[str, Any]]:
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    stats_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if not stats_path.is_file():
        raise FileNotFoundError(f"LeTools normalizer state not found: {stats_path}")
    stats = load_file(str(stats_path), device="cpu")
    normalizer = LinearNormalizer()
    normalizer["agent_pos"] = _minmax_field(stats, "observation.state")
    normalizer["action"] = _minmax_field(stats, "action")
    return normalizer, stats


class LeToolsObservationEncoder(nn.Module):
    """Use LeTools' RGB encoder and the checkpoint's exact preprocessing stats."""

    def __init__(
        self,
        rgb_encoder: nn.Module,
        image_keys: list[str],
        stats: dict[str, torch.Tensor],
        state_dim: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.image_keys = tuple(image_keys)
        self.state_dim = int(state_dim)
        self.eps = float(eps)
        self.image_feature_dim = int(rgb_encoder.feature_dim)
        self.feature_dim = self.state_dim + self.image_feature_dim * len(self.image_keys)
        for index, key in enumerate(self.image_keys):
            self.register_buffer(f"image_mean_{index}", stats[f"{key}.mean"].to(torch.float32))
            self.register_buffer(f"image_std_{index}", stats[f"{key}.std"].to(torch.float32))

    def output_shape(self) -> tuple[int]:
        return (self.feature_dim,)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if "agent_pos" not in obs:
            raise KeyError("RL-100 observation is missing 'agent_pos'.")
        features = [obs["agent_pos"].to(torch.float32)]
        for index, key in enumerate(self.image_keys):
            if key not in obs:
                raise KeyError(f"RL-100 observation is missing camera key {key!r}.")
            image = obs[key]
            if image.dtype == torch.uint8:
                image = image.to(torch.float32) / 255.0
            else:
                image = image.to(torch.float32)
            mean = getattr(self, f"image_mean_{index}").reshape(1, -1, 1, 1)
            std = getattr(self, f"image_std_{index}").reshape(1, -1, 1, 1)
            image = (image - mean) / (std + self.eps)
            features.append(self.rgb_encoder(image))
        return torch.cat(features, dim=-1)


class LeToolsUnetAdapter(nn.Module):
    """Add RL-100's keyword signature around LeRobot's unchanged U-Net."""

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_cond is not None:
            raise ValueError("LeTools Diffusion Policy does not use local conditioning.")
        return self.unet(sample, timestep, global_cond=global_cond)


class LeToolsDPPolicyAdapter(nn.Module):
    """A LeTools actor that satisfies the RL-100 BPPO policy contract."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "cuda:0",
        num_inference_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.source_checkpoint = str(Path(checkpoint_dir).expanduser().resolve())
        config = load_diffusion_config_rl100(self.source_checkpoint)
        config.device = device
        loaded_policy = DiffusionPolicy.from_pretrained(
            self.source_checkpoint,
            config=config,
            strict=True,
            local_files_only=True,
        )
        normalizer, stats = load_checkpoint_normalizer(self.source_checkpoint)

        image_keys = list(config.image_features)
        state_dim = int(config.robot_state_feature.shape[0])
        self.obs_encoder = LeToolsObservationEncoder(
            loaded_policy.diffusion.rgb_encoder,
            image_keys=image_keys,
            stats=stats,
            state_dim=state_dim,
        )
        self.model = LeToolsUnetAdapter(loaded_policy.diffusion.unet)
        self.normalizer = normalizer
        del loaded_policy

        self.noise_scheduler = DDPMSchedulerWithLogProb(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        checkpoint_inference_steps = int(
            config.num_inference_steps or config.num_train_timesteps
        )
        self.num_inference_steps = int(
            checkpoint_inference_steps
            if num_inference_steps is None
            else num_inference_steps
        )
        if not 1 <= self.num_inference_steps <= int(config.num_train_timesteps):
            raise ValueError(
                "num_inference_steps must be between 1 and "
                f"{config.num_train_timesteps}, got {self.num_inference_steps}."
            )
        self.horizon = int(config.horizon)
        self.n_obs_steps = int(config.n_obs_steps)
        self.n_action_steps = int(config.n_action_steps)
        self.action_dim = int(config.action_feature.shape[0])
        self.obs_feature_dim = int(self.obs_encoder.output_shape()[0])
        self.global_cond_dim = self.obs_feature_dim * self.n_obs_steps
        self.no_pre_action = False
        self.start = self.n_obs_steps - 1
        self.end = self.start + self.n_action_steps
        self.action_norm = True
        self.chunk_as_single_action = True
        self.obs_as_global_cond = True
        self.condition_type = "film"
        self.joint_opt_encoder = False
        self.use_pc_color = True
        self.w_pc = False
        self.is_flow = False
        self.eta = 1.0
        self.to(device)
        # BPPO deep-copies this adapter into a frozen behavior policy and a
        # trainable policy. Initialize the scheduler before that copy so both
        # policies can evaluate denoising-transition log probabilities.
        self._set_scheduler_timesteps()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def get_unet_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps

    def get_dynamics_encoder(self) -> nn.Module:
        return deepcopy(self.obs_encoder)

    def _set_scheduler_timesteps(self) -> None:
        try:
            self.noise_scheduler.set_timesteps(
                self.num_inference_steps, device=self.device
            )
        except TypeError:
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

    def obs2feature(
        self,
        obs_dict: dict[str, torch.Tensor],
        training: bool = False,
        fix_encoder: bool = True,
    ):
        del training
        normalized = self.normalizer.normalize(obs_dict)
        batch_size = normalized["agent_pos"].shape[0]
        flattened = {
            key: value[:, : self.n_obs_steps].reshape(-1, *value.shape[2:]).to(self.device)
            for key, value in normalized.items()
        }
        if fix_encoder:
            self.obs_encoder.eval()
            with torch.no_grad():
                encoded = self.obs_encoder(flattened)
        else:
            encoded = self.obs_encoder(flattened)
        encoded = encoded.reshape(batch_size, self.n_obs_steps, -1)
        global_cond = encoded.reshape(batch_size, -1)
        cond_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        return cond_data, cond_mask, None, global_cond, global_cond

    def feature2cond(self, features: torch.Tensor, batch_size: int):
        global_cond = features.reshape(batch_size, -1).to(self.device)
        cond_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        return cond_data, cond_mask, None, global_cond, global_cond

    def obs2latent(self, obs_dict: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        return self.obs2feature(obs_dict, fix_encoder=True)[-1]

    def all_step_logprob(
        self,
        state_dict: dict[str, torch.Tensor],
        fix_encoder: bool = True,
        training: bool = False,
    ):
        with torch.no_grad():
            cond_data, cond_mask, local_cond, global_cond, features = self.obs2feature(
                state_dict, training=training, fix_encoder=fix_encoder
            )
            trajectory = torch.randn_like(cond_data)
            self._set_scheduler_timesteps()
            all_x: list[torch.Tensor] = []
            all_logprob: list[torch.Tensor] = []
            for timestep in self.noise_scheduler.timesteps:
                trajectory[cond_mask] = cond_data[cond_mask]
                batch_timesteps = timestep.reshape(1).to(self.device).expand(trajectory.shape[0])
                model_output = self.model(
                    sample=trajectory,
                    timestep=self.get_unet_timesteps(batch_timesteps),
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
                all_x.append(trajectory)
                scheduler_output, log_prob = self.noise_scheduler.step_logprob(
                    model_output, batch_timesteps, trajectory
                )
                trajectory = scheduler_output.prev_sample
                all_logprob.append(log_prob)
            trajectory[cond_mask] = cond_data[cond_mask]
            all_x.append(trajectory)
        return (
            all_x[:-1],
            all_x[1:],
            all_logprob,
            cond_data,
            cond_mask,
            local_cond,
            global_cond,
            features,
        )

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cond_data, cond_mask, local_cond, global_cond, _ = self.obs2feature(obs_dict)
        trajectory = torch.randn_like(cond_data)
        self._set_scheduler_timesteps()
        for timestep in self.noise_scheduler.timesteps:
            batch_timesteps = timestep.reshape(1).to(self.device).expand(trajectory.shape[0])
            model_output = self.model(
                sample=trajectory,
                timestep=batch_timesteps,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output, timestep, trajectory
            ).prev_sample
        action_pred = self.normalizer["action"].unnormalize(trajectory)
        return {
            "action": action_pred[:, self.start : self.end],
            "action_pred": action_pred,
        }
