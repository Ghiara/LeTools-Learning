"""Export a BPPO-updated actor as a normal LeTools/LeRobot checkpoint bundle."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from .policy_adapter import load_diffusion_config_rl100


def _cpu_contiguous_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in module.state_dict().items()
    }


def export_letools_checkpoint(
    policy_adapter,
    source_checkpoint: str | Path,
    destination: str | Path,
    *,
    bppo_step: int,
    run_name: str,
) -> Path:
    """Copy the source bundle and replace only encoder/U-Net actor weights."""
    source = Path(source_checkpoint).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source LeTools checkpoint does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint directory: {output}")

    temporary = output.with_name(output.name + ".tmp_rl100")
    if temporary.exists():
        raise FileExistsError(
            f"Stale temporary export exists; inspect and remove it manually: {temporary}"
        )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)

    config = load_diffusion_config_rl100(source)
    config.device = "cpu"
    target = DiffusionPolicy.from_pretrained(
        str(source), config=config, strict=True, local_files_only=True
    )
    target.diffusion.unet.load_state_dict(
        policy_adapter.model.unet.state_dict(), strict=True
    )
    target.diffusion.rgb_encoder.load_state_dict(
        policy_adapter.obs_encoder.rgb_encoder.state_dict(), strict=True
    )
    save_file(
        _cpu_contiguous_state_dict(target),
        str(temporary / "model.safetensors"),
        metadata={"format": "pt"},
    )
    del target

    metadata = {
        "run_name": run_name,
        "bppo_step": int(bppo_step),
        "source_bc_checkpoint": str(source),
        "exported_at": datetime.now().astimezone().isoformat(),
        "format": "LeRobot DiffusionPolicy pretrained_model",
    }
    (temporary / "offline_bppo_metadata_rl100.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.rename(output)
    return output
