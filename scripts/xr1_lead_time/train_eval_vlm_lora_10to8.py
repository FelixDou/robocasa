"""Train ONE rank-16 pure-PyTorch LoRA adapter on Qwen3-VL-4B-Instruct strictly
using the 10 pre-registered Train/Dev tasks, then score ALL 8 Held-Out Unseen
Composition Test-1 tasks strictly zero-shot.

Saves:
  - ``<output>/lora_adapter_train10.pt`` (reusable for LLM-generated task-graph scoring!)
  - ``<output>/vlm_scores/vlm_destination_scores.npz`` + ``vlm_destination_index.json``
    (covering all 8 held-out tasks with atomic per-rollout resume!)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np

from . import analyze_recovery_pilot6 as pilot6
from .extract_vlm_destinations import (
    DEFAULT_QWEN3_VL_DIR,
    Qwen3VLTernaryScorer,
    _decode_video_pil_views,
    compute_ternary_scores,
    expand_causal_zoh,
    sampled_decision_positions,
    task_destination_predicates,
)
from .train_eval_vlm_lora_cross_task import (
    _load_rollout_records,
    _train_one_fold,
    format_binary_question,
    inject_lora_adapters,
    lora_state_dict,
    select_balanced_training_samples,
)


def run_train10_to_heldout8_lora(
    roots: list[Path],
    output_dir: Path,
    model_dir: Path,
    *,
    train_tasks: tuple[str, ...] = pilot6.BENCHMARK18_TRAIN_TASKS,
    heldout_tasks: tuple[str, ...] = pilot6.BENCHMARK18_HELDOUT_TASKS,
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
    seed: int = 17,
) -> dict:
    import torch

    train_set = set(train_tasks)
    held_set = set(heldout_tasks)
    if train_set & held_set:
        raise ValueError(f"Train/held-out overlap: {train_set & held_set}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "vlm_scores"
    ckpt_dir = scores_dir / "per_rollout"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "lora_adapter_train10.pt"

    t_all = time.time()
    records, input_hashes = _load_rollout_records(roots)
    by_rid = {r["rollout_id"]: r for r in records}
    held_recs = [r for r in records if r["task"] in held_set]
    train_recs = [r for r in records if r["task"] in train_set]
    if not train_recs or not held_recs:
        raise RuntimeError(f"Missing rollouts: train={len(train_recs)}, heldout={len(held_recs)}")
    print(
        f"Loaded {len(train_recs)} train rollouts ({len({r['task'] for r in train_recs})} tasks) "
        f"and {len(held_recs)} held-out rollouts ({len({r['task'] for r in held_recs})} tasks)"
    )

    # Check if all held-out checkpoints already exist (resume-safe)
    missing_held = [r for r in held_recs if not (ckpt_dir / f"{r['rollout_id']}.json").exists()]
    train_log = None
    if missing_held or not adapter_path.exists():
        scorer = Qwen3VLTernaryScorer(model_dir)
        lora_modules = inject_lora_adapters(
            scorer.model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            include_vision=include_vision,
        )
        n_tr = sum(p.numel() for p in scorer.model.parameters() if p.requires_grad)
        print(f"Injected {len(lora_modules)} LoRA adapters (trainable={n_tr:,})")

        if adapter_path.exists():
            ckpt = torch.load(str(adapter_path), map_location=scorer.device)
            scorer.model.load_state_dict(ckpt["state_dict"], strict=False)
            scorer.model.eval()
            print(f"Resumed trained LoRA adapter from {adapter_path}")
        else:
            samps = select_balanced_training_samples(records, train_set, seed=seed)
            # Strict leakage guard: assert ZERO held-out tasks in training samples
            if any(s["task"] in held_set for s in samps):
                raise RuntimeError("Leakage detected: held-out task present in training samples!")
            n_pos = sum(1 for s in samps if s["pred_label"] == 1)
            print(f"Selected {len(samps)} balanced train samples from 10 train tasks (pos={n_pos}, neg={len(samps)-n_pos})")

            # Pre-decode unique training frames once
            needed = defaultdict(set)
            for s in samps:
                needed[s["rollout_id"]].add(s["frame_idx"])
            t_dec = time.time()
            frame_cache = {}
            for rid, fset in sorted(needed.items()):
                decoded = _decode_video_pil_views(by_rid[rid]["video_path"], np.asarray(sorted(fset), dtype=int))
                for fi, views in decoded.items():
                    frame_cache[(rid, int(fi))] = views
            print(f"Pre-decoded {len(frame_cache)} unique training frames in {time.time() - t_dec:.1f}s")

            if scorer.device.type == "cuda":
                vram_gb = torch.cuda.get_device_properties(scorer.device).total_memory / (1024 ** 3)
                if vram_gb < 40.0:
                    scorer.model.enable_input_require_grads()
                    scorer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            train_log = _train_one_fold(
                scorer,
                samps,
                frame_cache,
                epochs=epochs,
                micro_batch_size=micro_batch_size,
                grad_accum=grad_accum,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                fold_idx=0,
            )
            torch.save(
                {
                    "train_tasks": list(train_tasks),
                    "heldout_tasks": list(heldout_tasks),
                    "rank": rank,
                    "alpha": alpha,
                    "include_vision": include_vision,
                    "state_dict": lora_state_dict(scorer.model),
                },
                adapter_path,
            )
            print(f"Saved 10-train LoRA adapter -> {adapter_path}")
            del frame_cache

        # Evaluate strictly zero-shot on all missing held-out rollouts
        t_ev = time.time()
        ev_q = 0
        for idx_r, rec in enumerate(missing_held):
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
            ev_q += len(query_items)

            pred_payload = {}
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
                }
            ep_scores = {
                "rollout_id": rid,
                "task": task,
                "task_instruction": rec["task_instruction"],
                "num_decisions": n_dec,
                "decision_stride": int(decision_stride),
                "sampled_positions": sampled_pos.tolist(),
                "predicates": pred_payload,
            }
            tmp_ckpt = ckpt_dir / f".{rid}.json.tmp"
            tmp_ckpt.write_text(json.dumps(ep_scores) + "\n")
            tmp_ckpt.replace(ckpt_dir / f"{rid}.json")

            if (idx_r + 1) % 20 == 0 or (idx_r + 1) == len(missing_held):
                el = max(1e-6, time.time() - t_ev)
                print(
                    f"[heldout eval {idx_r + 1:3d}/{len(missing_held)}] {task:21s} {rid[:8]} | "
                    f"q={ev_q} ({ev_q / el:.1f} q/s) | elapsed={el:.1f}s"
                )

    # Pack all held-out rollout scores into standard NPZ + index
    episodes_meta = []
    packed_arrays = {}
    for rec in held_recs:
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
                "task_instruction": rec["task_instruction"],
                "success": rec["success"],
                "num_decisions": n_dec,
                "sampled_positions": sampled_pos.tolist(),
                "predicates": pred_names,
            }
        )

    index_payload = {
        "schema_version": 1,
        "protocol": "train10_to_heldout8_lora_zero_shot",
        "train_tasks": list(train_tasks),
        "heldout_tasks": list(heldout_tasks),
        "model_dir": str(model_dir),
        "adapter_path": str(adapter_path),
        "decision_stride": int(decision_stride),
        "train_log": train_log,
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

    print(f"Done 10-train -> 8-heldout LoRA in {time.time() - t_all:.1f}s -> {scores_dir}")
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    run_train10_to_heldout8_lora(
        args.run,
        args.output,
        args.model_dir,
        decision_stride=args.decision_stride,
        eval_batch_size=args.eval_batch_size,
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
