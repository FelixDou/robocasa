"""Zero-shot Qwen3-VL-4B-Instruct destination & condition verifier over synchronized
RoboCasa365 recovery rollouts.

Reads the synchronized videos and manifests produced by
``scripts/run_xr1_recovery_pilot_6tasks.sh`` and computes single-forward-pass
``Yes`` / ``No`` / ``Unknown`` log-probabilities for every destination and hold
predicate in ``analyze_recovery_pilot6.TASK_SPECS`` using strictly causal
video-frame alignment (``frame_step <= decision_step``) and strictly non-privileged
prompts (derived solely from the observable ``task_instruction`` and predicate text).

Per-rollout checkpoints are saved atomically under ``<output>/per_rollout/`` so
runs are safely resumable, and the final merged output is written to
``<output>/vlm_destination_scores.npz`` + ``<output>/vlm_destination_index.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from . import analyze_recovery_pilot6 as pilot6
from .extract_perception_features import causal_frame_indices, split_obs_grid_3cam


DEFAULT_QWEN3_VL_DIR = (
    "/gs/bs/tga-shinoda/felid/robocasa_checkpoints/lingbot_vla_v2/downloads/Qwen3-VL-4B-Instruct"
)

# Strictly non-privileged visual verification questions for every placement target
# and hold condition in pilot6.TASK_SPECS.
# ``invert=True`` asks the positive visual presence question (avoiding double-negation
# confusion in VLMs) and flips Yes <-> No logits afterward.
PREDICATE_PROMPTS: dict[str, dict[str, dict[str, Any]]] = {
    "ArrangeBreadBasket": {
        "bread_in_basket": {
            "question": (
                "Is the bread currently resting inside the basket "
                "(not on the cabinet shelf and not held in the air by the robot)?"
            ),
            "invert": False,
        },
        "basket_on_dining_counter": {
            "question": (
                "Is the basket currently resting on the dining counter or dining table "
                "(away from the main kitchen counter under the wall cabinet)?"
            ),
            "invert": False,
        },
    },
    "CuttingToolSelection": {
        "correct_tool_present_on_board": {
            "question": (
                "Task: {task_instruction} "
                "Between a knife and a peeler, is the appropriate cutting tool for this task "
                "currently resting on top of the cutting board "
                "(not inside the drawer and not held in the air)?"
            ),
            "invert": False,
        },
        "wrong_tool_not_on_cutting_board": {
            "question": (
                "Task: {task_instruction} "
                "Between a knife and a peeler, is the WRONG (inappropriate) cutting tool "
                "for this task currently resting on top of the cutting board?"
            ),
            "invert": True,
        },
    },
    "DeliverStraw": {
        "straw_in_glass": {
            "question": (
                "Is a drinking straw currently placed inside the glass cup on the counter "
                "(not inside the drawer and not held in the robot gripper)?"
            ),
            "invert": False,
        },
    },
    "GarnishPancake": {
        "strawberry_on_pancake": {
            "question": (
                "Is a strawberry currently resting on top of the pancake on the plate?"
            ),
            "invert": False,
        },
        "pancake_on_plate": {
            "question": "Is the pancake currently resting on the plate?",
            "invert": False,
        },
        "plate_on_table": {
            "question": (
                "Is the plate with the pancake currently resting upright on the dining counter or table?"
            ),
            "invert": False,
        },
    },
    "MakeIceLemonade": {
        "lemon_in_glass": {
            "question": (
                "Is a yellow lemon wedge currently inside the glass cup of lemonade?"
            ),
            "invert": False,
        },
        "ice_in_glass": {
            "question": (
                "Is an ice cube currently inside the glass cup of lemonade?"
            ),
            "invert": False,
        },
    },
    "SetUpCuttingStation": {
        "knife_on_cutting_board": {
            "question": (
                "Is the knife currently resting on top of the cutting board "
                "(not inside the drawer and not held in the robot gripper)?"
            ),
            "invert": False,
        },
        "meat_on_cutting_board": {
            "question": (
                "Is the piece of meat currently resting on top of the cutting board "
                "(not on the plate and not held in the robot gripper)?"
            ),
            "invert": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are a precise visual state verifier for a kitchen manipulation robot. "
    "You are shown three synchronized camera views from the exact same moment: "
    "a left third-person robot view, a right third-person robot view, and a close-up wrist camera view. "
    "Answer the question strictly based on what is visually present right now."
)

ANSWER_INSTRUCTION = (
    "Answer with a single word: Yes, No, or Unknown "
    "(answer Unknown only if the relevant object and destination are both out of view in all three cameras)."
)


def task_destination_predicates(task: str) -> list[str]:
    """Return deterministic ordered list of destination/hold predicates for ``task``."""
    spec = pilot6.TASK_SPECS[task]
    ordered = []
    for _, _, tp in spec["placements"]:
        if tp not in ordered:
            ordered.append(tp)
    for hc in spec["hold_conditions"]:
        if hc not in ordered:
            ordered.append(hc)
    return ordered


def format_predicate_question(task: str, predicate: str, task_instruction: str) -> tuple[str, bool]:
    """Format the non-privileged visual question and return ``(prompt_text, invert)``."""
    spec = PREDICATE_PROMPTS[task][predicate]
    q = spec["question"].format(task_instruction=(task_instruction or "").strip())
    return f"{q}\n{ANSWER_INSTRUCTION}", bool(spec["invert"])


def sampled_decision_positions(num_decisions: int, decision_stride: int) -> np.ndarray:
    """Return strictly sorted unique decision indices to evaluate with the VLM.

    Always includes index ``0`` and the final decision ``num_decisions - 1``.
    """
    if num_decisions <= 0:
        return np.array([], dtype=int)
    stride = max(1, int(decision_stride))
    pos = list(range(0, num_decisions, stride))
    if pos[-1] != num_decisions - 1:
        pos.append(num_decisions - 1)
    return np.asarray(pos, dtype=int)


def expand_causal_zoh(
    num_decisions: int,
    sampled_positions: np.ndarray,
    sampled_values: np.ndarray,
) -> np.ndarray:
    """Expand sampled decision values to all ``num_decisions`` via strictly causal zero-order hold.

    For every decision index ``i``, outputs the value at the latest sampled index ``s <= i``.
    Never reads any sampled index ``s > i``.
    """
    if num_decisions == 0:
        return np.zeros((0,), dtype=np.float32)
    pos = np.asarray(sampled_positions, dtype=int)
    vals = np.asarray(sampled_values, dtype=np.float32)
    if len(pos) == 0 or pos[0] != 0:
        raise ValueError("sampled_positions must be non-empty and start at 0 for causal ZOH")
    if len(pos) != len(vals):
        raise ValueError("sampled_positions and sampled_values length mismatch")
    out = np.empty(num_decisions, dtype=np.float32)
    for j, start in enumerate(pos):
        end = int(pos[j + 1]) if j + 1 < len(pos) else num_decisions
        out[int(start) : end] = vals[j]
    return out


def compute_ternary_scores(
    logit_yes: np.ndarray,
    logit_no: np.ndarray,
    logit_unk: np.ndarray,
    *,
    invert: bool = False,
) -> dict[str, np.ndarray]:
    """Compute binary Yes-vs-No logit diff, conditional prob, and visibility prob.

    When ``invert=True``, swaps ``logit_yes`` and ``logit_no`` (used for negated
    hold predicates like ``wrong_tool_not_on_cutting_board`` asked positively).
    """
    ly = np.asarray(logit_yes, dtype=np.float64)
    ln = np.asarray(logit_no, dtype=np.float64)
    lu = np.asarray(logit_unk, dtype=np.float64)
    if invert:
        ly, ln = ln, ly

    logit_diff = ly - ln
    # Numerically stable sigmoid(logit_diff)
    prob_yes_given_yn = np.where(
        logit_diff >= 0.0,
        1.0 / (1.0 + np.exp(-np.clip(logit_diff, -40.0, 40.0))),
        np.exp(np.clip(logit_diff, -40.0, 40.0)) / (1.0 + np.exp(np.clip(logit_diff, -40.0, 40.0))),
    )
    # 3-way softmax over (Yes, No, Unknown) for visibility confidence p_vis = 1 - P(Unknown)
    stacked = np.stack([ly, ln, lu], axis=-1)
    m = stacked.max(axis=-1, keepdims=True)
    ex = np.exp(np.clip(stacked - m, -60.0, 0.0))
    probs3 = ex / ex.sum(axis=-1, keepdims=True)
    p_unk = probs3[..., 2]
    p_vis = 1.0 - p_unk

    return {
        "logit_diff": logit_diff.astype(np.float32),
        "prob_yes": prob_yes_given_yn.astype(np.float32),
        "prob_visible": p_vis.astype(np.float32),
        "prob_unknown": p_unk.astype(np.float32),
        "logit_yes": ly.astype(np.float32),
        "logit_no": ln.astype(np.float32),
        "logit_unk": lu.astype(np.float32),
    }


def _collect_single_token_ids(tokenizer, candidates: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for text in candidates:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1 and int(enc[0]) not in ids:
            ids.append(int(enc[0]))
    if not ids:
        raise RuntimeError(f"Tokenizer produced no single-token IDs for candidates {candidates}")
    return ids


def _last_nonpad_token_indices(attention_mask):
    """Exact index of the last non-pad token per batch item for both left and right padding."""
    import torch

    seq_len = attention_mask.shape[1]
    pos = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
    return (attention_mask.long() * pos).argmax(dim=1)


def _decode_video_pil_views(
    video_path: Path,
    frame_indices: np.ndarray,
) -> dict[int, tuple[Image.Image, Image.Image, Image.Image]]:
    """Decode only the needed frames sequentially and return RGB PIL ``(left, right, wrist)``."""
    needed = set(int(x) for x in frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    decoded: dict[int, tuple[Image.Image, Image.Image, Image.Image]] = {}
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in needed:
                l_rgb, r_rgb, w_rgb = split_obs_grid_3cam(bgr)
                decoded[idx] = (
                    Image.fromarray(l_rgb),
                    Image.fromarray(r_rgb),
                    Image.fromarray(w_rgb),
                )
            idx += 1
    finally:
        cap.release()
    missing = needed - set(decoded)
    if missing:
        raise RuntimeError(f"Video {video_path} missing requested frames: {sorted(missing)[:5]}")
    return decoded


class Qwen3VLTernaryScorer:
    """Single-forward-pass batched Yes/No/Unknown logprob scorer for Qwen3-VL-Instruct."""

    def __init__(self, model_dir: Path, device=None, max_pixels: int = 256 * 256):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model_dir = Path(model_dir)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        # Keep native 256x256 camera crops without unintended upscaling.
        if hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            cur_max = getattr(ip, "max_pixels", None)
            if isinstance(cur_max, int) and cur_max > max_pixels:
                ip.max_pixels = max_pixels
            cur_min = getattr(ip, "min_pixels", None)
            if isinstance(cur_min, int) and cur_min > max_pixels:
                ip.min_pixels = max_pixels
            size_cfg = getattr(ip, "size", None)
            if isinstance(size_cfg, dict) and isinstance(size_cfg.get("longest_edge"), int):
                size_cfg["longest_edge"] = min(size_cfg["longest_edge"], max_pixels)
                if isinstance(size_cfg.get("shortest_edge"), int):
                    size_cfg["shortest_edge"] = min(size_cfg["shortest_edge"], max_pixels)
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model = ModelCls.from_pretrained(
            str(self.model_dir),
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device).eval()

        tok = self.processor.tokenizer
        self.yes_ids = _collect_single_token_ids(tok, ("Yes", "yes", " Yes", " yes"))
        self.no_ids = _collect_single_token_ids(tok, ("No", "no", " No", " no"))
        self.unk_ids = _collect_single_token_ids(tok, ("Unknown", "unknown", " Unknown", " unknown"))
        print(
            f"Loaded {ModelCls.__name__} from {self.model_dir} on {self.device} ({self.dtype}) "
            f"in {time.time() - t0:.1f}s | yes_ids={self.yes_ids} no_ids={self.no_ids} unk_ids={self.unk_ids}"
        )

    def _build_messages(self, views: tuple[Image.Image, Image.Image, Image.Image], question: str) -> list[dict]:
        left_im, right_im, wrist_im = views
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left view:"},
                    {"type": "image", "image": left_im},
                    {"type": "text", "text": "Right view:"},
                    {"type": "image", "image": right_im},
                    {"type": "text", "text": "Wrist view:"},
                    {"type": "image", "image": wrist_im},
                    {"type": "text", "text": question},
                ],
            },
        ]

    def _prepare_batch(self, items: list[tuple[tuple[Image.Image, Image.Image, Image.Image], str]]):
        batch_msgs = [self._build_messages(views, question) for views, question in items]
        try:
            return self.processor.apply_chat_template(
                batch_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            texts = [
                self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_msgs
            ]
            flat_images = [im for views, _ in items for im in views]
            return self.processor(
                text=texts,
                images=flat_images,
                padding=True,
                return_tensors="pt",
            )

    def score_batch(
        self,
        items: list[tuple[tuple[Image.Image, Image.Image, Image.Image], str]],
        batch_size: int = 16,
        *,
        log_shapes: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(logit_yes, logit_no, logit_unk)`` arrays of shape ``(len(items),)``."""
        import torch

        yes_out, no_out, unk_out = [], [], []
        yes_t = torch.tensor(self.yes_ids, device=self.device, dtype=torch.long)
        no_t = torch.tensor(self.no_ids, device=self.device, dtype=torch.long)
        unk_t = torch.tensor(self.unk_ids, device=self.device, dtype=torch.long)

        with torch.inference_mode():
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                inputs = self._prepare_batch(chunk)
                if log_shapes and start == 0:
                    shapes = {
                        k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")
                    }
                    grid = inputs.get("image_grid_thw")
                    grid_info = grid[:3].tolist() if grid is not None else None
                    print(f"Preflight tensor shapes: {shapes} | image_grid_thw[:3]={grid_info}")
                inputs = {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
                outputs = self.model(**inputs, use_cache=False)
                logits = outputs.logits  # (B, T, V)
                last_idx = _last_nonpad_token_indices(inputs["attention_mask"])
                last_logits = logits[
                    torch.arange(logits.shape[0], device=self.device), last_idx, :
                ].float()
                ly = torch.logsumexp(last_logits.index_select(1, yes_t), dim=1)
                ln = torch.logsumexp(last_logits.index_select(1, no_t), dim=1)
                lu = torch.logsumexp(last_logits.index_select(1, unk_t), dim=1)
                yes_out.append(ly.cpu().numpy())
                no_out.append(ln.cpu().numpy())
                unk_out.append(lu.cpu().numpy())
                del inputs, outputs, logits, last_logits

        return (
            np.concatenate(yes_out, axis=0).astype(np.float32),
            np.concatenate(no_out, axis=0).astype(np.float32),
            np.concatenate(unk_out, axis=0).astype(np.float32),
        )


def _discover_manifests(roots: list[Path]) -> list[Path]:
    manifests = []
    for r in roots:
        if (r / "manifest.jsonl").exists():
            manifests.append(r / "manifest.jsonl")
        else:
            manifests.extend(sorted(r.glob("shard*/manifest.jsonl")))
    if not manifests:
        raise FileNotFoundError(f"No manifest.jsonl under {[str(r) for r in roots]}")
    return sorted(manifests)


def extract_vlm_scores(
    roots: list[Path],
    output_dir: Path,
    model_dir: Path,
    *,
    decision_stride: int = 2,
    batch_size: int = 16,
    max_rollouts_per_task: int | None = None,
    scorer: Any = None,
) -> dict:
    """Run zero-shot Qwen3-VL destination verification across rollouts with per-rollout resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "per_rollout"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = _discover_manifests(roots)
    rows_with_base: list[tuple[Path, dict]] = []
    seen_ids = set()
    input_hashes = {}
    per_task_counts: dict[str, int] = {}

    for mp in manifest_paths:
        input_hashes[str(mp)] = hashlib.sha256(mp.read_bytes()).hexdigest()
        base = mp.parent
        for line in mp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["rollout_id"]
            if rid in seen_ids:
                raise ValueError(f"Duplicate rollout_id: {rid}")
            task = row["task_name"]
            if max_rollouts_per_task is not None and per_task_counts.get(task, 0) >= max_rollouts_per_task:
                continue
            seen_ids.add(rid)
            per_task_counts[task] = per_task_counts.get(task, 0) + 1
            rows_with_base.append((base, row))

    if scorer is None:
        scorer = Qwen3VLTernaryScorer(model_dir)

    # Preflight sanity check on 1 synthetic batch before looping over videos.
    if hasattr(scorer, "score_batch"):
        dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        kwargs = {"batch_size": 1}
        if isinstance(scorer, Qwen3VLTernaryScorer):
            kwargs["log_shapes"] = True
        ly0, ln0, lu0 = scorer.score_batch(
            [((dummy, dummy, dummy), "Is there a cup?\n" + ANSWER_INSTRUCTION)],
            **kwargs,
        )
        if not (np.isfinite(ly0[0]) and np.isfinite(ln0[0]) and np.isfinite(lu0[0])):
            raise RuntimeError(f"Non-finite preflight logits: {(ly0, ln0, lu0)}")
        print(
            f"Preflight VLM check OK: yes={float(ly0[0]):.2f} no={float(ln0[0]):.2f} unk={float(lu0[0]):.2f}"
        )

    episodes_meta = []
    packed_arrays: dict[str, np.ndarray] = {}
    t_start = time.time()
    total_queries = 0
    computed_rollouts = 0
    resumed_rollouts = 0

    for ep_idx, (base, row) in enumerate(rows_with_base):
        rid = row["rollout_id"]
        task = row["task_name"]
        pred_names = task_destination_predicates(task)
        task_instruction = str(row.get("task_instruction") or "")
        trace = json.loads((base / row["subtask_trace_path"]).read_text())
        inf_records = trace["inference_records"]
        dec_steps = np.array([int(x["environment_step"]) for x in inf_records], dtype=int)
        n_dec = len(dec_steps)
        sampled_pos = sampled_decision_positions(n_dec, decision_stride)

        ckpt_path = ckpt_dir / f"{rid}.json"
        if ckpt_path.exists():
            ep_scores = json.loads(ckpt_path.read_text())
            if (
                ep_scores.get("decision_stride") == int(decision_stride)
                and ep_scores.get("sampled_positions") == sampled_pos.tolist()
                and set(ep_scores.get("predicates", {}).keys()) == set(pred_names)
            ):
                resumed_rollouts += 1
            else:
                ep_scores = None
        else:
            ep_scores = None

        if ep_scores is None:
            f_idx_all = causal_frame_indices(dec_steps, trace["video_frame_environment_steps"])
            f_idx_sampled = f_idx_all[sampled_pos]
            decoded_views = _decode_video_pil_views(base / row["video_path"], f_idx_sampled)

            # Build flat list of (views, question) for all (sampled_pos, predicate) in this rollout.
            query_items = []
            query_meta = []
            for p_name in pred_names:
                q_text, invert = format_predicate_question(task, p_name, task_instruction)
                for s_i, fidx in enumerate(f_idx_sampled):
                    query_items.append((decoded_views[int(fidx)], q_text))
                    query_meta.append((p_name, s_i, invert))

            ly, ln, lu = scorer.score_batch(query_items, batch_size=batch_size)
            total_queries += len(query_items)
            computed_rollouts += 1

            pred_payload: dict[str, dict[str, list[float]]] = {}
            n_s = len(sampled_pos)
            cursor = 0
            for p_name in pred_names:
                _, invert = format_predicate_question(task, p_name, task_instruction)
                sl = slice(cursor, cursor + n_s)
                cursor += n_s
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
                "task_instruction": task_instruction,
                "num_decisions": n_dec,
                "decision_stride": int(decision_stride),
                "sampled_positions": sampled_pos.tolist(),
                "sampled_decision_steps": [int(dec_steps[p]) for p in sampled_pos],
                "sampled_frame_indices": [int(x) for x in f_idx_sampled],
                "predicates": pred_payload,
            }
            tmp_ckpt = ckpt_dir / f".{rid}.json.tmp"
            tmp_ckpt.write_text(json.dumps(ep_scores) + "\n")
            tmp_ckpt.replace(ckpt_path)

            if computed_rollouts in (1, 2, 5) or (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == len(rows_with_base):
                elapsed = max(1e-6, time.time() - t_start)
                qps = total_queries / elapsed
                rem = len(rows_with_base) - (ep_idx + 1)
                eta = (elapsed / max(1, computed_rollouts)) * rem
                print(
                    f"[{ep_idx + 1:3d}/{len(rows_with_base)}] {task:21s} {rid[:8]} | "
                    f"queries={total_queries} ({qps:.1f} q/s) | elapsed={elapsed:.1f}s ETA={eta:.1f}s"
                )

        # Pack causal ZOH dense arrays + exact sampled arrays into NPZ keyed by rollout_id__predicate
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
                "task_instruction": task_instruction,
                "success": not bool(row["failed"]),
                "num_decisions": n_dec,
                "sampled_positions": sampled_pos.tolist(),
                "predicates": pred_names,
            }
        )

    index_payload = {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "decision_stride": int(decision_stride),
        "batch_size": int(batch_size),
        "num_episodes": len(episodes_meta),
        "resumed_rollouts": resumed_rollouts,
        "computed_rollouts": computed_rollouts,
        "total_queries_computed": total_queries,
        "elapsed_sec": round(time.time() - t_start, 2),
        "inputs_sha256": input_hashes,
        "episodes": episodes_meta,
    }

    with TemporaryDirectory(dir=output_dir) as tmp:
        staged_npz = Path(tmp) / "vlm_destination_scores.npz"
        staged_json = Path(tmp) / "vlm_destination_index.json"
        np.savez_compressed(staged_npz, **packed_arrays)
        staged_json.write_text(json.dumps(index_payload, indent=2) + "\n")
        staged_npz.replace(output_dir / "vlm_destination_scores.npz")
        staged_json.replace(output_dir / "vlm_destination_index.json")

    print(
        f"Done: {len(episodes_meta)} rollouts (computed={computed_rollouts}, resumed={resumed_rollouts}, "
        f"queries={total_queries}) in {time.time() - t_start:.1f}s -> {output_dir}"
    )
    return index_payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_QWEN3_VL_DIR))
    parser.add_argument("--decision-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rollouts-per-task", type=int, default=None)
    args = parser.parse_args(argv)
    extract_vlm_scores(
        args.run,
        args.output,
        args.model_dir,
        decision_stride=args.decision_stride,
        batch_size=args.batch_size,
        max_rollouts_per_task=args.max_rollouts_per_task,
    )


if __name__ == "__main__":
    main()
