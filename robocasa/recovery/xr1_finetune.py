"""Fine-tune the served XR-1 (RoboCasa365) model with its own flow-matching forward.

Step 0 of ``docs/recovery_harvest_finetune_plan.md``, decision A (26 November 2026). The released
Xiaomi trainer trains a different XR-1 variant (extra choice/score heads, the state in the VLM, a
+10 DiT position shift, no video input), so this module trains the model the server runs:

- the model is loaded from the Hugging Face folder exactly as ``deploy/server.py`` loads it;
- the VLM runs as in ``MiBoTForActionGeneration.forward``, frozen and without gradients;
- the DiT is called through the model's own ``dit_forward`` on a noised action chunk, with the
  same DiT positions and attention mask as the served forward (no +10 shift);
- the loss is the flow-matching part of the released ``XR1.py``: t = 0.999 (1 - b),
  b ~ Beta(1.5, 1), x_t = (1 - t) noise + t action, target action - noise, and
  0.5 MSE + 1.0 FFT-magnitude loss over the active action dims (0-11, from the processor's
  ``action_mask``). No choice/score heads, no async prefix;
- the trainable set is the action expert (``TRAINABLE_MODULES``); parameters stay in bfloat16 as
  served, with float32 master weights in the optimizer;
- checkpoints are written as a derived Hugging Face folder (trained weights plus every non-weight
  file of a template folder, normally the SAFE-patched checkpoint), so every collector serves them
  unchanged.

Subcommands (cluster; PyTorch, transformers 4.57.1 and the checkpoint's remote code):

- ``input-match``: the Step 0 gate. On held-out demo frames, the sample builder and the deployed
  client (``XiaomiRobotics1Policy._make_request``) must give identical processor outputs.
- ``loss-check``: descriptive. The loaded weights' loss on held-out demos, next to the same loss
  with action chunks taken from another episode of the same task.
- ``train``: single node, ``torchrun --nproc_per_node N``.
- ``diff-weights``: per-module differences between two Hugging Face folders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from robocasa.recovery import xr1_samples as xs
from robocasa.recovery.expert_segments import PILOT_TASKS, lerobot_dir

# Frozen with this code (Step 0, decision A).
TRAINABLE_MODULES = ("dit", "state_projector", "action_projector", "action_output_layer", "t_embedder",
                     "t_projector", "sink")
FLOW_BETA_A = 1.5            # b ~ Beta(1.5, 1): inverse CDF b = u ** (1 / 1.5)
FLOW_T_SCALE = 0.999
MSE_WEIGHT = 0.5             # released XR1.py: 0.5 * loss_mse + freq_coefficient (1.0) * loss_freq
FREQ_WEIGHT = 1.0
# Optimizer and schedule of the released trainer (configs/trainer/deepspeed.yaml).
LR_START, LR_MAX, LR_MIN, WARMUP_STEPS = 5e-7, 2e-5, 5e-6, 500
BETAS, WEIGHT_DECAY, EPS, GRAD_CLIP = (0.9, 0.95), 0.1, 1e-8, 1.0
WEIGHT_FILE_SUFFIXES = (".safetensors",)
WEIGHT_INDEX = "model.safetensors.index.json"
MANIFEST = "xr1_finetune_manifest.json"


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except Exception:
        return None


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- model --------------------------------------------------------------------------------------

def load_model(model_dir, device="cuda"):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True,
                                      attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    return model.to(device).eval()


def load_processor(model_dir):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False,
                                         local_files_only=True)


def prepare_trainable(model, modules=TRAINABLE_MODULES):
    """Freeze everything, then unfreeze the action expert. Parameters keep their served dtype."""
    for param in model.parameters():
        param.requires_grad_(False)
    params = []
    for name in modules:
        for param in getattr(model, name).parameters():
            param.requires_grad_(True)
            params.append(param)
    return params


def collate(samples, pad_token_id):
    """Batch processor outputs: right-pad the text, concatenate the videos in sample order."""
    import torch

    lengths = [int(s["input_ids"].shape[1]) for s in samples]
    width = max(lengths)
    input_ids = torch.full((len(samples), width), int(pad_token_id), dtype=samples[0]["input_ids"].dtype)
    attention_mask = torch.zeros((len(samples), width), dtype=samples[0]["attention_mask"].dtype)
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        input_ids[i, :length] = sample["input_ids"][0]
        attention_mask[i, :length] = sample["attention_mask"][0]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": torch.cat([s["pixel_values_videos"] for s in samples]),
        "video_grid_thw": torch.cat([s["video_grid_thw"] for s in samples]),
        "state": torch.cat([torch.as_tensor(s["state"]) for s in samples]),
        "action_mask": torch.cat([torch.as_tensor(s["action_mask"]) for s in samples]),
        "action": torch.cat([torch.as_tensor(s["action"], dtype=torch.float32) for s in samples]),
        "meta": [s.get("meta") for s in samples],
    }


def to_device(batch, device):
    """As ``deploy/server.py``: floating tensors to bfloat16, the target stays float32."""
    import torch

    out = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
        elif key == "action":
            out[key] = value.to(device=device, dtype=torch.float32)
        elif value.is_floating_point():
            out[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            out[key] = value.to(device=device)
    return out


def vlm_context(model, batch):
    """The served forward's VLM pass (no logits): the KV cache and the 3-D RoPE positions."""
    import torch

    with torch.no_grad():
        out = model.vlm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              pixel_values_videos=batch["pixel_values_videos"],
                              video_grid_thw=batch["video_grid_thw"], use_cache=True)
    return out.past_key_values, out.position_ids


def dit_inputs(model, position_ids, attention_mask, state, action_mask):
    """DiT positions and attention mask, as ``MiBoTForActionGeneration.forward`` builds them."""
    import torch

    batch, action_length, _ = action_mask.shape
    query = action_length + state.shape[1] + 1
    positions = (torch.arange(0, query, device=action_mask.device).view(1, 1, -1).repeat(3, batch, 1)
                 + position_ids.max(dim=-1)[0][..., None] + 1)
    position_embeds = model.rotary_emb(action_mask, positions)
    dit_mask = torch.tril(torch.ones((batch, query, query), device=action_mask.device), diagonal=0)
    cache_mask = attention_mask[:, None, :].expand(-1, query, -1)
    attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
    return position_embeds, attn_mask


def flow_timesteps(u):
    """t = 0.999 (1 - b), b ~ Beta(1.5, 1) by its inverse CDF, from uniforms ``u``."""
    return (1.0 - u.float() ** (1.0 / FLOW_BETA_A)) * FLOW_T_SCALE


def flow_loss(model, batch, *, u=None, noise=None, per_sample=False):
    """Flow-matching loss of the action expert on one batch (``to_device`` already applied)."""
    import torch

    cache, position_ids = vlm_context(model, batch)
    action_mask, state = batch["action_mask"], batch["state"]
    action = batch["action"].float()
    size = action.shape[0]
    position_embeds, attn_mask = dit_inputs(model, position_ids, batch["attention_mask"], state, action_mask)
    state_embed = model.state_projector(state)
    if u is None:
        u = torch.rand(size, device=action.device)
    t = flow_timesteps(u).to(action_mask.dtype)[:, None, None]
    if noise is None:
        noise = torch.randn_like(action)
    noisy = (1 - t.float()) * noise + t.float() * action
    target = action - noise
    pred = model.dit_forward(noisy_action=noisy.to(action_mask.dtype), t=t, action_mask=action_mask,
                             state_embed=state_embed, position_embeds=position_embeds,
                             past_key_values=cache, attn_mask=attn_mask).float()
    mask = action_mask.bool()
    squared = (pred - target) ** 2
    freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
    freq_mask = mask[:, : freq.shape[1]]
    if per_sample:
        mse = (squared * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
        loss_freq = (freq * freq_mask).sum(dim=(1, 2)) / freq_mask.sum(dim=(1, 2))
        return MSE_WEIGHT * mse + FREQ_WEIGHT * loss_freq, {"mse": mse, "freq": loss_freq}
    mse = squared[mask].mean()
    loss_freq = freq[freq_mask].mean()
    return MSE_WEIGHT * mse + FREQ_WEIGHT * loss_freq, {"mse": mse.detach(), "freq": loss_freq.detach()}


def learning_rate(step, max_steps, *, start=LR_START, peak=LR_MAX, end=LR_MIN, warmup=WARMUP_STEPS):
    """Linear warmup from ``start`` to ``peak``, then cosine down to ``end`` at ``max_steps``."""
    if warmup > 0 and step < warmup:
        return start + (peak - start) * step / warmup
    span = max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * progress))


class MasterAdamW:
    """AdamW on float32 master copies of bfloat16 parameters, with cross-rank gradient averaging."""

    def __init__(self, params, *, lr=LR_START, betas=BETAS, weight_decay=WEIGHT_DECAY, eps=EPS):
        import torch

        self.params = list(params)
        self.master = [p.detach().float().clone().requires_grad_(True) for p in self.params]
        decay = [m for m in self.master if m.ndim >= 2]
        no_decay = [m for m in self.master if m.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas, eps=eps)

    def step(self, lr, clip=GRAD_CLIP):
        import torch
        import torch.distributed as dist
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        flat = _flatten_dense_tensors([g.float() for g in grads])
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(flat)
            flat /= dist.get_world_size()
        for master, grad in zip(self.master, _unflatten_dense_tensors(flat, self.master)):
            master.grad = grad
        norm = torch.nn.utils.clip_grad_norm_(self.master, clip)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for param, master in zip(self.params, self.master):
                param.copy_(master)
                param.grad = None
        return float(norm)


def save_derived_folder(model, template_dir, out_dir, manifest):
    """Write the model's weights plus every non-weight file of ``template_dir``; check the key set."""
    template_dir, out_dir = Path(template_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; derived folders are never overwritten")
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="5GB")
    for path in template_dir.iterdir():
        if path.is_file() and path.suffix not in WEIGHT_FILE_SUFFIXES and path.name != WEIGHT_INDEX:
            shutil.copy2(path, tmp / path.name)
    expected = set(json.loads((template_dir / WEIGHT_INDEX).read_text())["weight_map"])
    written = set(json.loads((tmp / WEIGHT_INDEX).read_text())["weight_map"])
    if expected != written:
        raise RuntimeError(f"weight keys differ from the template: missing {sorted(expected - written)[:5]}, "
                           f"extra {sorted(written - expected)[:5]}")
    (tmp / MANIFEST).write_text(json.dumps({**manifest, "template_dir": str(template_dir),
                                            "saved_at": _utc_now()}, indent=2))
    tmp.rename(out_dir)
    return out_dir


# ---- data ---------------------------------------------------------------------------------------

def episode_list(tasks, dataset_base, split, max_episodes_per_task=None, lerobot_overrides=None):
    """(task, LeRobot folder, episode) for the demos of ``split`` ("train" or "heldout")."""
    out = []
    for task in tasks:
        folder = lerobot_dir(task, dataset_base, lerobot_overrides)
        episodes = [e for e in xs.lerobot_episodes(folder) if xs.episode_split(task, e) == split]
        if max_episodes_per_task is not None:
            episodes = episodes[: int(max_episodes_per_task)]
        if not episodes:
            raise ValueError(f"{task}: no {split} episodes under {folder}")
        out.extend((task, str(folder), e) for e in episodes)
    return out


def make_stream(episodes, processor_dir, *, samples_per_episode, buffer_size, seed, rank):
    """Infinite shuffled stream of training samples (one ``IterableDataset`` per DataLoader)."""
    import torch

    class EpisodeStream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            worker = 0 if info is None else info.id
            rng = np.random.default_rng([int(seed), int(rank), int(worker)])
            processor = load_processor(processor_dir)
            buffer = []
            while True:
                task, folder, episode = episodes[int(rng.integers(len(episodes)))]
                ep = xs.read_lerobot_episode(folder, episode, task)
                times = list(xs.valid_times(ep.num_frames))
                if not times:
                    continue
                for t in rng.choice(times, size=min(samples_per_episode, len(times)), replace=False):
                    buffer.append(xs.make_sample(processor, ep, int(t)))
                while len(buffer) >= buffer_size:
                    yield buffer.pop(int(rng.integers(len(buffer))))

    return EpisodeStream()


# ---- subcommands --------------------------------------------------------------------------------

def run_input_match(args):
    processor = load_processor(args.model_dir)
    rng = np.random.default_rng(args.seed)
    rows, per_task = [], {}
    for task, folder, episode in _pick(episode_list(args.tasks, args.dataset_base, "heldout"), args, rng):
        ep = xs.read_lerobot_episode(folder, episode, task)
        times = list(xs.valid_times(ep.num_frames))
        for t in sorted(rng.choice(times, size=min(args.frames_per_episode, len(times)), replace=False)):
            mismatched = xs.compare_with_policy(processor, ep, int(t))
            rows.append({"task_name": task, "episode": episode, "t": int(t), "mismatched": mismatched})
            stats = per_task.setdefault(task, {"samples": 0, "identical": 0})
            stats["samples"] += 1
            stats["identical"] += int(not mismatched)
    passed = all(s["samples"] >= args.min_samples_per_task and s["identical"] == s["samples"]
                 for s in per_task.values()) and set(per_task) == set(args.tasks)
    result = {"kind": "xr1_input_match", "model_dir": str(args.model_dir), "passed": passed,
              "rule": f"every sample identical, at least {args.min_samples_per_task} per task",
              "per_task": per_task, "git_commit": _git_commit(), "created_at": _utc_now()}
    _write_result(args.output_dir, "input_match", result, rows)
    print(json.dumps({"passed": passed, "per_task": per_task}, indent=2))
    return 0 if passed else 1


def _pick(episodes, args, rng):
    by_task = {}
    for item in episodes:
        by_task.setdefault(item[0], []).append(item)
    picked = []
    for task in args.tasks:
        items = by_task.get(task, [])
        order = rng.permutation(len(items))[: args.episodes_per_task]
        picked.extend(items[i] for i in sorted(order))
    return picked


def _write_result(output_dir, name, result, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    with (output_dir / f"{name}_samples.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def run_loss_check(args):
    import torch

    torch.manual_seed(args.seed)
    model = load_model(args.model_dir)
    processor = load_processor(args.model_dir)
    pad = processor.tokenizer.pad_token_id
    rng = np.random.default_rng(args.seed)
    picked = _pick(episode_list(args.tasks, args.dataset_base, "heldout"), args, rng)
    episodes = {}
    for task, folder, episode in picked:
        episodes.setdefault(task, []).append(xs.read_lerobot_episode(folder, episode, task))
    rows = []
    for task, eps in episodes.items():
        if len(eps) < 2:
            raise ValueError(f"{task}: the loss check needs at least two held-out episodes")
        for i, ep in enumerate(eps):
            times = list(xs.valid_times(ep.num_frames))
            for t in sorted(rng.choice(times, size=min(args.frames_per_episode, len(times)), replace=False)):
                other = eps[(i + 1 + int(rng.integers(len(eps) - 1))) % len(eps)]
                other_t = int(rng.choice(list(xs.valid_times(other.num_frames))))
                own = xs.make_sample(processor, ep, int(t))
                swapped = dict(own, action=xs.sample_arrays(other, other_t)[2])
                batch = to_device(collate([own, swapped], pad), "cuda")
                generator = torch.Generator(device="cuda").manual_seed(int(rng.integers(2**31)))
                u = torch.rand(1, device="cuda", generator=generator).repeat(2)
                noise = torch.randn((1, *batch["action"].shape[1:]), device="cuda", generator=generator).repeat(2, 1, 1)
                with torch.no_grad():
                    loss, parts = flow_loss(model, batch, u=u, noise=noise, per_sample=True)
                rows.append({"task_name": task, "episode": ep.episode, "t": int(t), "other_episode": other.episode,
                             "other_t": other_t, "loss_own": float(loss[0]), "loss_other": float(loss[1]),
                             "mse_own": float(parts["mse"][0]), "mse_other": float(parts["mse"][1])})
    per_task = {}
    for task in episodes:
        task_rows = [r for r in rows if r["task_name"] == task]
        per_task[task] = {
            "samples": len(task_rows),
            "loss_own_mean": float(np.mean([r["loss_own"] for r in task_rows])),
            "loss_other_mean": float(np.mean([r["loss_other"] for r in task_rows])),
            "own_below_other_share": float(np.mean([r["loss_own"] < r["loss_other"] for r in task_rows])),
        }
    result = {"kind": "xr1_loss_check", "model_dir": str(args.model_dir), "descriptive_only": True,
              "per_task": per_task, "git_commit": _git_commit(), "created_at": _utc_now()}
    _write_result(args.output_dir, "loss_check", result, rows)
    print(json.dumps(per_task, indent=2))
    return 0


def run_train(args):
    import torch
    import torch.distributed as dist

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_dir, device)
    params = prepare_trainable(model)
    optimizer = MasterAdamW(params, weight_decay=args.weight_decay)
    processor = load_processor(args.model_dir)
    episodes = episode_list(args.tasks, args.dataset_base, "train", args.max_episodes_per_task)
    stream = make_stream(episodes, args.model_dir, samples_per_episode=args.samples_per_episode,
                         buffer_size=args.buffer_size, seed=args.seed, rank=rank)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=lambda samples: collate(samples, processor.tokenizer.pad_token_id),
        persistent_workers=args.num_workers > 0, prefetch_factor=2 if args.num_workers > 0 else None)
    manifest = {"kind": "xr1_finetune", "source_model_dir": str(args.model_dir), "tasks": list(args.tasks),
                "trainable_modules": list(TRAINABLE_MODULES), "world_size": world,
                "trainable_parameters": int(sum(p.numel() for p in params)), "episodes": len(episodes),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"},
                "git_commit": _git_commit()}
    if rank == 0:
        (output_dir / "run.json").write_text(json.dumps({**manifest, "started_at": _utc_now()}, indent=2))
        print(json.dumps({k: manifest[k] for k in ("world_size", "trainable_parameters", "episodes")}), flush=True)

    iterator = iter(loader)
    tic = time.time()
    for step in range(1, args.max_steps + 1):
        batch = to_device(next(iterator), device)
        loss, parts = flow_loss(model, batch)
        loss.backward()
        lr = learning_rate(step, args.max_steps, start=args.lr_start, peak=args.lr_max, end=args.lr_min,
                           warmup=args.warmup_steps)
        norm = optimizer.step(lr, args.grad_clip)
        if rank == 0 and (step % args.log_every == 0 or step == 1 or step == args.max_steps):
            elapsed = time.time() - tic
            row = {"step": step, "loss": float(loss.detach()), "mse": float(parts["mse"]),
                   "freq": float(parts["freq"]), "lr": lr, "grad_norm": norm,
                   "samples_per_second": step * args.batch_size * world / elapsed,
                   "max_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30, "elapsed_s": elapsed}
            with (output_dir / "metrics.jsonl").open("a") as stream_out:
                stream_out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            if distributed:
                dist.barrier()
            if rank == 0:
                saved = save_derived_folder(model, args.template_dir or args.model_dir,
                                            output_dir / "checkpoints" / f"step_{step:06d}",
                                            {**manifest, "step": step})
                print(f"SAVED {saved}", flush=True)
            if distributed:
                dist.barrier()
    if distributed:
        dist.destroy_process_group()
    return 0


def run_diff_weights(args):
    import torch
    from safetensors import safe_open

    def tensors(folder):
        folder = Path(folder)
        index = json.loads((folder / WEIGHT_INDEX).read_text())["weight_map"]
        for name in sorted(set(index.values())):
            with safe_open(str(folder / name), framework="pt") as handle:
                for key in handle.keys():
                    yield key, handle.get_tensor(key)

    reference = dict(tensors(args.a))
    groups = {}
    for key, value in tensors(args.b):
        group = key.split(".")[0]
        diff = (value.float() - reference.pop(key).float()).abs()
        stats = groups.setdefault(group, {"tensors": 0, "changed_tensors": 0, "max_abs_diff": 0.0})
        stats["tensors"] += 1
        stats["changed_tensors"] += int(bool(torch.any(diff > 0)))
        stats["max_abs_diff"] = max(stats["max_abs_diff"], float(diff.max()) if diff.numel() else 0.0)
    result = {"a": str(args.a), "b": str(args.b), "only_in_a": sorted(reference), "groups": groups}
    print(json.dumps(result, indent=2))
    return 0 if not reference else 1


# ---- CLI ----------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--model-dir", type=Path, required=True, help="RoboCasa365 Hugging Face folder")
        p.add_argument("--dataset-base", type=Path, required=True, help="RoboCasa dataset root (ROBOCASA_DATASET_BASE)")
        p.add_argument("--tasks", nargs="+", default=list(PILOT_TASKS))
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("input-match", help="Step 0 gate: builder vs deployed client on held-out demo frames")
    data_args(p)
    p.add_argument("--episodes-per-task", type=int, default=5)
    p.add_argument("--frames-per-episode", type=int, default=4)
    p.add_argument("--min-samples-per-task", type=int, default=20)
    p.set_defaults(func=run_input_match)

    p = sub.add_parser("loss-check", help="descriptive: loss with the own vs another episode's actions")
    data_args(p)
    p.add_argument("--episodes-per-task", type=int, default=8)
    p.add_argument("--frames-per-episode", type=int, default=4)
    p.set_defaults(func=run_loss_check)

    p = sub.add_parser("train", help="train the action expert (torchrun --nproc_per_node N)")
    data_args(p)
    p.add_argument("--template-dir", type=Path, default=None,
                   help="non-weight files of saved folders (default: --model-dir; use the SAFE-patched folder)")
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8, help="per GPU")
    p.add_argument("--lr-start", type=float, default=LR_START)
    p.add_argument("--lr-max", type=float, default=LR_MAX)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    p.add_argument("--save-every", type=int, default=10**9)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--samples-per-episode", type=int, default=4)
    p.add_argument("--buffer-size", type=int, default=32)
    p.add_argument("--max-episodes-per-task", type=int, default=None)
    p.set_defaults(func=run_train)

    p = sub.add_parser("diff-weights", help="per-module weight differences between two folders")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.set_defaults(func=run_diff_weights)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
