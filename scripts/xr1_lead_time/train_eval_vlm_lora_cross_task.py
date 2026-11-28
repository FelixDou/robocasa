"""3-fold cross-task LoRA adaptation of Qwen3-VL-4B-Instruct for zero-shot
unseen-task spatial destination verification on RoboCasa365.

Protocol (strictly zero target-task leakage):
- The 6 pilot tasks are partitioned into 3 balanced disjoint 2-task held-out folds
  (``CROSS_TASK_FOLDS``), each holding out 1 multi-object task + 1 single-object task.
- For each fold ``f in {0, 1, 2}``:
  1. Reset all pure-PyTorch LoRA adapters (no external ``peft`` dependency) cleanly
     to zero on the frozen ``Qwen3-VL-4B-Instruct`` backbone.
  2. Sample class-balanced, hard-post-release-negative-enriched ``(3-view frame,
     question, label)`` examples strictly from the 4 training tasks (80 rollouts).
  3. Train with direct binary cross-entropy on the exact single-forward-pass
     ``logit_yes - logit_no`` decision statistic (1-to-1 with inference; accounts
     for ``invert=True`` on negated hold predicates).
  4. Evaluate **strictly zero-shot** on all 40 rollouts of the 2 held-out unseen
     tasks at ``decision_stride = 2`` (causal frame mapping ``frame_step <= t``),
     writing atomic per-rollout checkpoints under ``<output>/vlm_scores/per_rollout/``.
- Merges all 120 unseen-task rollout predictions into the exact
  ``vlm_destination_scores.npz`` + ``vlm_destination_index.json`` schema consumed
  by ``eval_vlm_hybrid_monitor.py`` and ``eval_vlm_scoring_upgrades.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
from PIL import Image

from . import analyze_recovery_pilot6 as pilot6
from .extract_perception_features import causal_frame_indices
from .extract_vlm_destinations import (
    DEFAULT_QWEN3_VL_DIR,
    PREDICATE_PROMPTS,
    SYSTEM_PROMPT,
    Qwen3VLTernaryScorer,
    _decode_video_pil_views,
    _discover_manifests,
    _last_nonpad_token_indices,
    compute_ternary_scores,
    expand_causal_zoh,
    sampled_decision_positions,
    task_destination_predicates,
)


# Balanced 3-fold cross-task split: each fold holds out 1 multi-object task + 1 single-object task.
CROSS_TASK_FOLDS: tuple[tuple[str, str], ...] = (
    ("ArrangeBreadBasket", "DeliverStraw"),
    ("CuttingToolSelection", "MakeIceLemonade"),
    ("GarnishPancake", "SetUpCuttingStation"),
)

BINARY_ANSWER_INSTRUCTION = "Answer with a single word: Yes or No."

LORA_TARGET_LEAF_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj")


def format_binary_question(task: str, predicate: str, task_instruction: str) -> tuple[str, bool]:
    """Format strictly non-privileged binary Yes/No visual verification question."""
    spec = PREDICATE_PROMPTS[task][predicate]
    q = spec["question"].format(task_instruction=(task_instruction or "").strip())
    return f"{q}\n{BINARY_ANSWER_INSTRUCTION}", bool(spec["invert"])


def validate_cross_task_folds(
    folds: tuple[tuple[str, ...], ...] = CROSS_TASK_FOLDS,
    expected_tasks: tuple[str, ...] | set[str] | None = None,
) -> dict[str, int]:
    """Verify folds are pairwise disjoint and cover ``expected_tasks`` (default: PILOT6_TASKS)."""
    task_to_fold: dict[str, int] = {}
    for f_idx, held in enumerate(folds):
        for task in held:
            if task in task_to_fold:
                raise ValueError(f"Duplicate task across folds: {task}")
            task_to_fold[task] = f_idx
    expected = set(pilot6.PILOT6_TASKS if expected_tasks is None else expected_tasks)
    if set(task_to_fold.keys()) != expected:
        raise ValueError(f"Folds {set(task_to_fold)} do not cover expected tasks {expected}")
    return task_to_fold


def _evenly_spaced(indices: np.ndarray, k: int) -> list[int]:
    if len(indices) == 0 or k <= 0:
        return []
    if len(indices) <= k:
        return [int(x) for x in indices]
    pick = np.linspace(0, len(indices) - 1, k).round().astype(int)
    seen: list[int] = []
    for p in pick:
        v = int(indices[p])
        if v not in seen:
            seen.append(v)
    return seen


def select_balanced_training_samples(
    rollout_records: list[dict],
    train_tasks: set[str],
    *,
    max_pos_per_rollout: int = 12,
    max_hard_neg_per_rollout: int = 8,
    max_easy_neg_per_rollout: int = 4,
    seed: int = 0,
) -> list[dict]:
    """Select balanced, hard-negative-enriched training samples strictly from ``train_tasks``.

    Excludes constant all-True background hold preconditions (``pancake_on_plate``,
    ``plate_on_table``) that have zero negative examples, while training on all
    placement targets plus ``wrong_tool_not_on_cutting_board`` (which has real negatives).
    Balances positive vs. negative examples globally per training task.
    """
    rng = np.random.default_rng(seed)
    by_task_pos: dict[str, list[dict]] = defaultdict(list)
    by_task_neg: dict[str, list[dict]] = defaultdict(list)

    for rec in rollout_records:
        task = rec["task"]
        if task not in train_tasks:
            continue
        spec = pilot6.TASK_SPECS[task]
        n_dec = rec["num_decisions"]
        dec_steps = rec["decision_steps"]
        f_idx_all = rec["frame_indices"]
        pred_dec = rec["predicates_at_decisions"]

        # Train on all placement targets + any hold predicate that actually varies in this rollout
        train_preds = [tp for _, _, tp in spec["placements"]]
        for hc in spec["hold_conditions"]:
            if hc not in train_preds:
                vals = pred_dec[hc]
                if any(vals) and not all(vals):
                    train_preds.append(hc)

        # Build any-grasp onset index for distinguishing hard post-grasp negatives vs easy pre-grasp negatives
        g_keys = [gp for _, gps, _ in spec["placements"] for gp in gps]
        any_g = np.zeros(n_dec, dtype=bool)
        for gk in g_keys:
            any_g |= np.asarray(pred_dec[gk], dtype=bool)
        g_on = np.flatnonzero(any_g)
        first_grasp_idx = int(g_on[0]) if g_on.size else n_dec

        for pname in train_preds:
            y_arr = np.asarray(pred_dec[pname], dtype=bool)
            q_text, invert = format_binary_question(task, pname, rec["task_instruction"])
            # Raw VLM question target: if invert=True, "Yes" means predicate is False!
            raw_yes_target = (~y_arr) if invert else y_arr

            pos_idx = np.flatnonzero(y_arr)
            neg_idx = np.flatnonzero(~y_arr)
            hard_neg_idx = neg_idx[neg_idx >= first_grasp_idx]
            easy_neg_idx = neg_idx[neg_idx < first_grasp_idx]

            # Prioritize transitions (first few positive frames right after arrival) + evenly spaced coverage
            chosen_pos = _evenly_spaced(pos_idx, max_pos_per_rollout)
            if pos_idx.size and int(pos_idx[0]) not in chosen_pos:
                chosen_pos = [int(pos_idx[0])] + chosen_pos[: max_pos_per_rollout - 1]
            chosen_hard = _evenly_spaced(hard_neg_idx, max_hard_neg_per_rollout)
            chosen_easy = _evenly_spaced(easy_neg_idx, max_easy_neg_per_rollout)

            for d_i in chosen_pos:
                by_task_pos[task].append(
                    {
                        "rollout_id": rec["rollout_id"],
                        "task": task,
                        "predicate": pname,
                        "decision_idx": int(d_i),
                        "frame_idx": int(f_idx_all[d_i]),
                        "question": q_text,
                        "yes_label": float(raw_yes_target[d_i]),
                        "pred_label": 1,
                    }
                )
            for d_i in chosen_hard + chosen_easy:
                by_task_neg[task].append(
                    {
                        "rollout_id": rec["rollout_id"],
                        "task": task,
                        "predicate": pname,
                        "decision_idx": int(d_i),
                        "frame_idx": int(f_idx_all[d_i]),
                        "question": q_text,
                        "yes_label": float(raw_yes_target[d_i]),
                        "pred_label": 0,
                    }
                )

    # Cap negatives per task to at most 1.5x positives (or min 80) so class balance stays clean across tasks.
    samples: list[dict] = []
    for task in sorted(train_tasks):
        pos_list = list(by_task_pos[task])
        neg_list = list(by_task_neg[task])
        max_neg = max(80, int(round(1.5 * max(1, len(pos_list)))))
        if len(neg_list) > max_neg:
            keep = rng.choice(len(neg_list), size=max_neg, replace=False)
            neg_list = [neg_list[int(i)] for i in keep]
        samples.extend(pos_list)
        samples.extend(neg_list)

    order = rng.permutation(len(samples))
    return [samples[int(i)] for i in order]


def inject_lora_adapters(
    model,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    include_vision: bool = True,
    target_leaf_names: tuple[str, ...] = LORA_TARGET_LEAF_NAMES,
):
    """Freeze ``model`` and replace target ``nn.Linear`` modules in-place with pure-PyTorch LoRA.

    Returns the list of injected ``LoRALinear`` wrappers so ``reset_lora()`` can
    cleanly zero all adapters between cross-task folds without reloading base weights.
    """
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int, lora_alpha: float, drop: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.r = int(r)
            self.scaling = float(lora_alpha) / float(r)
            self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
            dev = base.weight.device
            self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=torch.float32, device=dev))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.float32, device=dev))
            self.reset_lora()

        def reset_lora(self):
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

        def forward(self, x):
            base_out = self.base(x)
            x_in = self.drop(x).to(dtype=self.lora_A.dtype)
            delta = (x_in @ self.lora_A.t()) @ self.lora_B.t()
            return base_out + delta.to(dtype=base_out.dtype) * self.scaling

    for p in model.parameters():
        p.requires_grad_(False)

    injected = []
    # Collect replacements first so we never mutate named_modules() while iterating
    targets: list[tuple[Any, str, nn.Linear, str]] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in full_name:
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        is_visual = "visual" in full_name or "vision" in full_name
        is_merger = "merger" in full_name or "projector" in full_name
        if is_visual and not include_vision:
            continue
        if leaf in target_leaf_names or (include_vision and is_merger):
            parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            targets.append((parent, leaf, module, full_name))

    for parent, leaf, orig_linear, _full_name in targets:
        wrapped = LoRALinear(orig_linear, rank, alpha, dropout)
        setattr(parent, leaf, wrapped)
        injected.append(wrapped)

    if not injected:
        raise RuntimeError("No target nn.Linear modules matched for LoRA injection")
    return injected


def lora_state_dict(model) -> dict[str, Any]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.named_parameters()
        if v.requires_grad
    }


def _train_one_fold(
    scorer: Qwen3VLTernaryScorer,
    train_samples: list[dict],
    frame_cache: dict[tuple[str, int], tuple[Image.Image, Image.Image, Image.Image]],
    *,
    epochs: int = 2,
    micro_batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    seed: int = 0,
    fold_idx: int = 0,
) -> dict:
    import torch
    import torch.nn.functional as F

    params = [p for p in scorer.model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    total_micro_steps = epochs * math.ceil(len(train_samples) / micro_batch_size)
    total_opt_steps = max(1, math.ceil(total_micro_steps / grad_accum))

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    warmup_steps = max(1, int(0.1 * total_opt_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    yes_t = torch.tensor(scorer.yes_ids, device=scorer.device, dtype=torch.long)
    no_t = torch.tensor(scorer.no_ids, device=scorer.device, dtype=torch.long)

    rng = np.random.default_rng(seed + 100 * fold_idx)
    scorer.model.train()
    opt.zero_grad(set_to_none=True)

    t0 = time.time()
    opt_step = 0
    micro_step = 0
    running_loss = 0.0
    running_n = 0
    loss_history = []

    for ep in range(epochs):
        perm = rng.permutation(len(train_samples))
        for start in range(0, len(train_samples), micro_batch_size):
            batch_idx = perm[start : start + micro_batch_size]
            items = []
            labels = []
            for idx in batch_idx:
                s = train_samples[int(idx)]
                views = frame_cache[(s["rollout_id"], s["frame_idx"])]
                items.append((views, s["question"]))
                labels.append(s["yes_label"])

            inputs = scorer._prepare_batch(items)
            inputs = {
                k: (v.to(scorer.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            y_t = torch.tensor(labels, device=scorer.device, dtype=torch.float32)

            outputs = scorer.model(**inputs, use_cache=False)
            logits = outputs.logits
            last_idx = _last_nonpad_token_indices(inputs["attention_mask"])
            last_logits = logits[
                torch.arange(logits.shape[0], device=scorer.device), last_idx, :
            ].float()
            ly = torch.logsumexp(last_logits.index_select(1, yes_t), dim=1)
            ln = torch.logsumexp(last_logits.index_select(1, no_t), dim=1)
            loss = F.binary_cross_entropy_with_logits(ly - ln, y_t)

            (loss / float(grad_accum)).backward()
            running_loss += float(loss.detach().cpu()) * len(labels)
            running_n += len(labels)
            micro_step += 1

            del inputs, outputs, logits, last_logits, ly, ln, loss

            if micro_step % grad_accum == 0 or (start + micro_batch_size >= len(train_samples) and ep == epochs - 1):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step in (1, 10, 25) or opt_step % 50 == 0 or opt_step == total_opt_steps:
                    avg_l = running_loss / max(1, running_n)
                    loss_history.append({"step": opt_step, "loss": round(avg_l, 4)})
                    cur_lr = opt.param_groups[0]["lr"]
                    elapsed = time.time() - t0
                    print(
                        f"  [fold {fold_idx}] step {opt_step:3d}/{total_opt_steps} | "
                        f"bce_loss={avg_l:.4f} | lr={cur_lr:.2e} | elapsed={elapsed:.1f}s"
                    )
                    running_loss = 0.0
                    running_n = 0

    scorer.model.eval()
    return {
        "fold": fold_idx,
        "num_train_samples": len(train_samples),
        "num_trainable_params": int(n_trainable),
        "opt_steps": opt_step,
        "train_sec": round(time.time() - t0, 2),
        "loss_history": loss_history,
    }


def _load_rollout_records(roots: list[Path]) -> tuple[list[dict], dict[str, str]]:
    manifest_paths = _discover_manifests(roots)
    records = []
    seen = set()
    hashes = {}
    for mp in manifest_paths:
        hashes[str(mp)] = hashlib.sha256(mp.read_bytes()).hexdigest()
        base = mp.parent
        for line in mp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["rollout_id"]
            if rid in seen:
                raise ValueError(f"Duplicate rollout_id: {rid}")
            seen.add(rid)
            task = row["task_name"]
            trace = json.loads((base / row["subtask_trace_path"]).read_text())
            inf_records = trace["inference_records"]
            dec_steps = np.array([int(x["environment_step"]) for x in inf_records], dtype=int)
            n_steps = int(row["num_environment_steps"])
            hist = trace["recovery_observations"]
            pred_names = sorted(hist[0]["values"].keys())
            series = {p: pilot6.expand_recovery_series(hist, n_steps, p) for p in pred_names}
            pred_at_dec = {p: [bool(series[p][int(t)]) for t in dec_steps] for p in pred_names}
            f_idx_all = causal_frame_indices(dec_steps, trace["video_frame_environment_steps"])
            records.append(
                {
                    "rollout_id": rid,
                    "task": task,
                    "task_instruction": str(row.get("task_instruction") or ""),
                    "success": not bool(row["failed"]),
                    "video_path": base / row["video_path"],
                    "num_decisions": len(dec_steps),
                    "decision_steps": dec_steps,
                    "frame_indices": f_idx_all,
                    "predicates_at_decisions": pred_at_dec,
                }
            )
    return records, hashes


def run_cross_task_lora(
    roots: list[Path],
    output_dir: Path,
    model_dir: Path,
    *,
    decision_stride: int = 2,
    eval_batch_size: int = 16,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    epochs: int = 2,
    micro_batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    include_vision: bool = True,
    seed: int = 7,
) -> dict:
    import torch

    validate_cross_task_folds(CROSS_TASK_FOLDS)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "vlm_scores"
    ckpt_dir = scores_dir / "per_rollout"
    adapters_dir = output_dir / "lora_adapters"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    records, input_hashes = _load_rollout_records(roots)
    by_rid = {r["rollout_id"]: r for r in records}
    print(f"Loaded {len(records)} rollouts across {len({r['task'] for r in records})} tasks")

    # Pre-plan balanced training samples for all 3 folds and pre-decode ONLY needed training frames ONCE.
    fold_train_samples: dict[int, list[dict]] = {}
    needed_train_frames: dict[str, set[int]] = defaultdict(set)
    all_fold_tasks = {t for fold in CROSS_TASK_FOLDS for t in fold}
    for f_idx, held_tasks in enumerate(CROSS_TASK_FOLDS):
        held_set = set(held_tasks)
        train_set = all_fold_tasks - held_set
        samps = select_balanced_training_samples(records, train_set, seed=seed + f_idx)
        fold_train_samples[f_idx] = samps
        for s in samps:
            needed_train_frames[s["rollout_id"]].add(s["frame_idx"])

    # Determine which folds actually still need training/inference (supports clean resume!)
    folds_needed = []
    for f_idx, held_tasks in enumerate(CROSS_TASK_FOLDS):
        held_recs = [r for r in records if r["task"] in held_tasks]
        all_done = True
        for r in held_recs:
            cp = ckpt_dir / f"{r['rollout_id']}.json"
            if not cp.exists():
                all_done = False
                break
        if not all_done:
            folds_needed.append(f_idx)
        else:
            print(f"Fold {f_idx} (held={held_tasks}) already complete on disk; resuming")

    fold_logs = []
    if folds_needed:
        t_dec = time.time()
        frame_cache: dict[tuple[str, int], tuple[Image.Image, Image.Image, Image.Image]] = {}
        active_rids = set()
        for f_idx in folds_needed:
            for s in fold_train_samples[f_idx]:
                active_rids.add(s["rollout_id"])
        for rid in sorted(active_rids):
            rec = by_rid[rid]
            f_list = np.asarray(sorted(needed_train_frames[rid]), dtype=int)
            decoded = _decode_video_pil_views(rec["video_path"], f_list)
            for fi, views in decoded.items():
                frame_cache[(rid, int(fi))] = views
        print(
            f"Pre-decoded {len(frame_cache)} unique training frames across {len(active_rids)} videos "
            f"in {time.time() - t_dec:.1f}s"
        )

        scorer = Qwen3VLTernaryScorer(model_dir)
        lora_modules = inject_lora_adapters(
            scorer.model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            include_vision=include_vision,
        )
        n_trainable = sum(p.numel() for p in scorer.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in scorer.model.parameters())
        print(
            f"Injected {len(lora_modules)} pure-PyTorch LoRA adapters (rank={rank}, alpha={alpha}, "
            f"include_vision={include_vision}) | trainable={n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)"
        )

        # Enable gradient checkpointing automatically on smaller (<40GB) GPUs
        if scorer.device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(scorer.device).total_memory / (1024 ** 3)
            print(f"CUDA device VRAM: {vram_gb:.1f} GB")
            if vram_gb < 40.0:
                scorer.model.enable_input_require_grads()
                scorer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                print("Enabled gradient checkpointing (<40GB GPU)")

        for f_idx in folds_needed:
            held_tasks = CROSS_TASK_FOLDS[f_idx]
            held_set = set(held_tasks)
            print(f"\n=== Fold {f_idx}/2 | HELD-OUT UNSEEN TASKS: {held_tasks} ===")
            for m in lora_modules:
                m.reset_lora()
            if scorer.device.type == "cuda":
                torch.cuda.empty_cache()

            samps = fold_train_samples[f_idx]
            n_pos = sum(1 for s in samps if s["pred_label"] == 1)
            print(f"  Training samples (4 train tasks only): {len(samps)} (pos={n_pos}, neg={len(samps)-n_pos})")
            tlog = _train_one_fold(
                scorer,
                samps,
                frame_cache,
                epochs=epochs,
                micro_batch_size=micro_batch_size,
                grad_accum=grad_accum,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                fold_idx=f_idx,
            )
            adapter_path = adapters_dir / f"fold{f_idx}_lora.pt"
            torch.save(
                {
                    "fold": f_idx,
                    "held_out_tasks": list(held_tasks),
                    "rank": rank,
                    "alpha": alpha,
                    "include_vision": include_vision,
                    "state_dict": lora_state_dict(scorer.model),
                },
                adapter_path,
            )

            # Zero-shot evaluation strictly on the 2 held-out unseen tasks
            held_recs = [r for r in records if r["task"] in held_set]
            t_ev = time.time()
            fold_q = 0
            for idx_r, rec in enumerate(held_recs):
                rid = rec["rollout_id"]
                task = rec["task"]
                pred_names = task_destination_predicates(task)
                n_dec = rec["num_decisions"]
                sampled_pos = sampled_decision_positions(n_dec, decision_stride)
                f_idx_s = rec["frame_indices"][sampled_pos]
                decoded_views = _decode_video_pil_views(rec["video_path"], f_idx_s)

                query_items = []
                for p_name in pred_names:
                    q_text, _ = format_binary_question(task, p_name, rec["task_instruction"])
                    for fidx in f_idx_s:
                        query_items.append((decoded_views[int(fidx)], q_text))

                ly, ln, lu = scorer.score_batch(query_items, batch_size=eval_batch_size)
                fold_q += len(query_items)

                pred_payload: dict[str, dict[str, list[float]]] = {}
                n_s = len(sampled_pos)
                cur = 0
                for p_name in pred_names:
                    _, invert = format_binary_question(task, p_name, rec["task_instruction"])
                    sl = slice(cur, cur + n_s)
                    cur += n_s
                    sc = compute_ternary_scores(ly[sl], ln[sl], lu[sl], invert=invert)
                    pred_payload[p_name] = {
                        "sampled_logit_diff": [round(float(x), 5) for x in sc["logit_diff"]],
                        "sampled_prob_yes": [round(float(x), 5) for x in sc["prob_yes"]],
                        "sampled_prob_visible": [round(float(x), 5) for x in sc["prob_visible"]],
                        "sampled_logit_yes": [round(float(x), 5) for x in sc["logit_yes"]],
                        "sampled_logit_no": [round(float(x), 5) for x in sc["logit_no"]],
                        "sampled_logit_unk": [round(float(x), 5) for x in sc["logit_unk"]],
                    }

                ep_scores = {
                    "rollout_id": rid,
                    "task": task,
                    "held_out_fold": f_idx,
                    "task_instruction": rec["task_instruction"],
                    "num_decisions": n_dec,
                    "decision_stride": int(decision_stride),
                    "sampled_positions": sampled_pos.tolist(),
                    "sampled_decision_steps": [int(rec["decision_steps"][p]) for p in sampled_pos],
                    "sampled_frame_indices": [int(x) for x in f_idx_s],
                    "predicates": pred_payload,
                }
                tmp_ckpt = ckpt_dir / f".{rid}.json.tmp"
                tmp_ckpt.write_text(json.dumps(ep_scores) + "\n")
                tmp_ckpt.replace(ckpt_dir / f"{rid}.json")

                if (idx_r + 1) % 10 == 0 or (idx_r + 1) == len(held_recs):
                    el = max(1e-6, time.time() - t_ev)
                    print(
                        f"  [fold {f_idx} eval {idx_r + 1:2d}/{len(held_recs)}] {task:20s} {rid[:8]} | "
                        f"queries={fold_q} ({fold_q / el:.1f} q/s) | elapsed={el:.1f}s"
                    )

            tlog["eval_queries"] = fold_q
            tlog["eval_sec"] = round(time.time() - t_ev, 2)
            tlog["held_out_tasks"] = list(held_tasks)
            fold_logs.append(tlog)

    # Merge all 120 unseen-task rollout checkpoints into standard vlm_destination_scores.npz
    episodes_meta = []
    packed_arrays: dict[str, np.ndarray] = {}
    for rec in records:
        rid = rec["rollout_id"]
        task = rec["task"]
        pred_names = task_destination_predicates(task)
        ep_scores = json.loads((ckpt_dir / f"{rid}.json").read_text())
        n_dec = rec["num_decisions"]
        sampled_pos = np.asarray(ep_scores["sampled_positions"], dtype=int)
        for p_name in pred_names:
            pp = ep_scores["predicates"][p_name]
            s_ld = np.asarray(pp["sampled_logit_diff"], dtype=np.float32)
            s_py = np.asarray(pp["sampled_prob_yes"], dtype=np.float32)
            s_pv = np.asarray(pp["sampled_prob_visible"], dtype=np.float32)
            key = f"{rid}__{p_name}"
            packed_arrays[f"{key}__sampled_logit_diff"] = s_ld
            packed_arrays[f"{key}__sampled_prob_yes"] = s_py
            packed_arrays[f"{key}__sampled_prob_visible"] = s_pv
            packed_arrays[f"{key}__zoh_logit_diff"] = expand_causal_zoh(n_dec, sampled_pos, s_ld)
            packed_arrays[f"{key}__zoh_prob_yes"] = expand_causal_zoh(n_dec, sampled_pos, s_py)
            packed_arrays[f"{key}__zoh_prob_visible"] = expand_causal_zoh(n_dec, sampled_pos, s_pv)

        episodes_meta.append(
            {
                "rollout_id": rid,
                "task": task,
                "held_out_fold": ep_scores.get("held_out_fold"),
                "task_instruction": rec["task_instruction"],
                "success": rec["success"],
                "num_decisions": n_dec,
                "sampled_positions": sampled_pos.tolist(),
                "predicates": pred_names,
            }
        )

    index_payload = {
        "schema_version": 1,
        "protocol": "3fold_cross_task_lora_unseen_task_zero_shot",
        "cross_task_folds": [list(f) for f in CROSS_TASK_FOLDS],
        "model_dir": str(model_dir),
        "decision_stride": int(decision_stride),
        "lora_config": {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "epochs": epochs,
            "micro_batch_size": micro_batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "weight_decay": weight_decay,
            "include_vision": include_vision,
            "seed": seed,
        },
        "fold_logs": fold_logs,
        "num_episodes": len(episodes_meta),
        "elapsed_sec": round(time.time() - t_all, 2),
        "inputs_sha256": input_hashes,
        "episodes": episodes_meta,
    }

    with TemporaryDirectory(dir=scores_dir) as tmp:
        staged_npz = Path(tmp) / "vlm_destination_scores.npz"
        staged_json = Path(tmp) / "vlm_destination_index.json"
        np.savez_compressed(staged_npz, **packed_arrays)
        staged_json.write_text(json.dumps(index_payload, indent=2) + "\n")
        staged_npz.replace(scores_dir / "vlm_destination_scores.npz")
        staged_json.replace(scores_dir / "vlm_destination_index.json")

    print(
        f"Done: 3-fold cross-task LoRA scored all {len(episodes_meta)} unseen-task rollouts "
        f"in {time.time() - t_all:.1f}s -> {scores_dir}"
    )
    return index_payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_QWEN3_VL_DIR))
    parser.add_argument("--decision-stride", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--no-vision-lora", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    run_cross_task_lora(
        args.run,
        args.output,
        args.model_dir,
        decision_stride=args.decision_stride,
        eval_batch_size=args.eval_batch_size,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        include_vision=not args.no_vision_lora,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
