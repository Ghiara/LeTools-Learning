"""Train LeTools DP no-depth checkpoints with RL-100 offline BPPO."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "lerobot" / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "rl100" / "RL-100"))

from omegaconf import OmegaConf  # noqa: E402

from kuavo_model.rl100_bridge.dp_nodepth.bppo.policy_adapter import (  # noqa: E402
    LeToolsDPPolicyAdapter,
)
from kuavo_model.rl100_bridge.dp_nodepth.bppo.transition_dataset import (  # noqa: E402
    LeRobotDPNoDepthTransitionDataset,
)
from kuavo_model.rl100_bridge.dp_nodepth.bppo.train_workspace import (  # noqa: E402
    OfflineBPPODPNoDepthWorkspace,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "train"
    / "rl100_dp_nodepth"
    / "icra_task1"
    / "offline_bppo.yaml"
)


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_config(cfg) -> None:
    if str(cfg.task.name) != "icra_task1":
        raise ValueError("The published task adapter currently supports icra_task1 only")
    if int(cfg.task.state_dim) != 16 or int(cfg.task.action_dim) != 16:
        raise ValueError("The validated ICRA Task 1 dataset uses 16-D state/action")

    allowed = Path(cfg.paths.allowed_root).expanduser().resolve()
    configured_repo = Path(cfg.paths.repo_root).expanduser().resolve()
    if configured_repo != REPO_ROOT:
        raise ValueError(
            f"paths.repo_root must identify this checkout: {REPO_ROOT}, "
            f"got {configured_repo}"
        )
    checked = {
        "repo_root": cfg.paths.repo_root,
        "dataset_root": cfg.paths.dataset_root,
        "terminal_annotations": cfg.paths.terminal_annotations,
        "bc_checkpoint": cfg.paths.bc_checkpoint,
        "output_root": cfg.paths.output_root,
        "log_root": cfg.paths.log_root,
        "tmp_root": cfg.paths.tmp_root,
    }
    for name, value in checked.items():
        resolved = Path(value).expanduser().resolve()
        if not _is_below(resolved, allowed):
            raise ValueError(f"paths.{name} must stay below {allowed}, got {resolved}")

    required = {
        "LeTools repository": Path(cfg.paths.repo_root),
        "LeRobot dataset": Path(cfg.paths.dataset_root),
        "terminal annotations": Path(cfg.paths.terminal_annotations),
        "BC checkpoint": Path(cfg.paths.bc_checkpoint),
        "terminal exclusion audit": Path(
            str(cfg.paths.terminal_annotations) + ".exclusions.json"
        ),
    }
    for label, path in required.items():
        if not path.expanduser().exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    info_path = Path(cfg.paths.dataset_root) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected = int(cfg.data.expected_episode_count)
    if int(info["total_episodes"]) != expected:
        raise ValueError(
            f"Config expects {expected} episodes, dataset has {info['total_episodes']}"
        )


def _gpu_preflight(cfg) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(str(cfg.runtime.device))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    utilization = None
    try:
        device_index = 0 if device.index is None else device.index
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        utilization = int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    print(
        f"GPU preflight: free={free_gib:.1f}/{total_gib:.1f} GiB, "
        f"utilization={utilization if utilization is not None else 'unknown'}%"
    )
    if bool(cfg.runtime.enforce_gpu_idle):
        if free_gib < float(cfg.runtime.min_free_gpu_gib):
            raise RuntimeError("GPU free memory is below the configured minimum")
        if utilization is not None and utilization > int(
            cfg.runtime.max_gpu_utilization_percent
        ):
            raise RuntimeError("GPU utilization is above the configured maximum")


def _configure_runtime(cfg) -> Path:
    tmp_root = Path(cfg.paths.tmp_root).expanduser().resolve()
    log_root = Path(cfg.paths.log_root).expanduser().resolve()
    output_root = Path(cfg.paths.output_root).expanduser().resolve()
    for path in (tmp_root, log_root, output_root):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

    log_path = log_root / (
        "rl100_bppo_dp_nodepth_icra_task1_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        + ".log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.getLogger(__name__).info("Log file: %s", log_path)
    return log_path


def _make_dataset(cfg, policy: LeToolsDPPolicyAdapter):
    dataset = LeRobotDPNoDepthTransitionDataset(
        dataset_root=cfg.paths.dataset_root,
        dataset_repo_id=cfg.data.dataset_repo_id,
        terminal_annotation_path=cfg.paths.terminal_annotations,
        excluded_episode_indices=list(cfg.task.excluded_episode_indices),
        gamma=float(cfg.training.gamma),
        video_backend=cfg.data.video_backend,
    )
    dataset.set_normalizer(deepcopy(policy.normalizer).cpu())
    return dataset


def _check_only(cfg) -> None:
    device = str(cfg.runtime.device)
    policy = LeToolsDPPolicyAdapter(
        cfg.paths.bc_checkpoint,
        device=device,
        num_inference_steps=int(cfg.bppo.num_inference_steps),
    )
    dataset = _make_dataset(cfg, policy)
    sample_index = dataset.indices_for_episodes({dataset.eligible_episode_indices[0]})[0]
    sample = dataset[sample_index]

    from kuavo_model.rl100_bridge.dp_nodepth.bppo.critic import IQL_Q_V_no  # noqa: F401
    from kuavo_model.rl100_bridge.dp_nodepth.bppo.ensemble_dynamics import (  # noqa: F401
        EnsembleDynamics_batch,
    )
    from kuavo_model.rl100_bridge.dp_nodepth.bppo.bppo import (  # noqa: F401
        BehaviorProximalPolicyOptimization,
    )

    print("Preflight passed: LeRobot sample, LeTools DP, Critic, Dynamics, BPPO")
    print(
        {
            "obs": tuple(sample["obs"]["agent_pos"].shape),
            "action": tuple(sample["action"].shape),
            "next_obs": tuple(sample["next_obs"]["agent_pos"].shape),
            "reward": tuple(sample["reward"].shape),
            "not_done": tuple(sample["not_done"].shape),
            "return": tuple(sample["return"].shape),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(args.config).expanduser().resolve())
    _validate_config(cfg)
    _gpu_preflight(cfg)
    _configure_runtime(cfg)
    if args.check_only:
        _check_only(cfg)
        return 0

    final_checkpoint = OfflineBPPODPNoDepthWorkspace(cfg).run()
    print(f"Training complete: {final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
