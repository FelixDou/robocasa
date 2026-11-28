"""Automatically generate visual placement verification questions from raw
observable ``task_instruction`` strings using local ``Qwen3-VL-4B-Instruct``
(text-only, zero simulator predicates, zero manual task engineering) and score
held-out rollouts using those LLM-generated task graphs.

Enables the reviewer-proof ablation on the 8 held-out Test-1 tasks:
  hand-specified non-privileged prompts vs. 100% LLM-generated task graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np

from .extract_perception_features import causal_frame_indices
from .extract_vlm_destinations import (
    ANSWER_INSTRUCTION,
    DEFAULT_QWEN3_VL_DIR,
    Qwen3VLTernaryScorer,
    _decode_video_pil_views,
    _discover_manifests,
    compute_ternary_scores,
    expand_causal_zoh,
    sampled_decision_positions,
)


TASK_GRAPH_SYSTEM_PROMPT = (
    "You decompose kitchen robot manipulation instructions into concise, visually "
    "verifiable placement completion questions. Output ONLY valid JSON — no prose, "
    "no markdown code fences."
)

TASK_GRAPH_USER_TEMPLATE = (
    'Robot task instruction: "{task_instruction}"\n'
    "List the 1 to 4 core object-placement destination states required to complete this task "
    "(ignore opening/closing doors or turning knobs; focus on objects placed into or onto "
    "their destination containers, boards, burners, racks, sinks, trays, or appliances).\n"
    'Return a JSON array of objects, each with keys "id" (short snake_case) and '
    '"question" (a single Yes/No visual question asking whether the object is currently '
    "resting at its destination and not held in the robot gripper)."
)


def parse_llm_task_graph_response(raw_text: str, task_instruction: str) -> list[dict[str, str]]:
    """Parse JSON list of ``{id, question}`` from LLM text with robust cleanup & fallback."""
    txt = (raw_text or "").strip()
    # Strip optional markdown fences
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    # Extract first JSON array bracket span if extra text surrounds it
    m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
    parsed = None
    if m is not None:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = None

    items: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    if isinstance(parsed, list):
        for idx, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                continue
            q = str(entry.get("question") or "").strip()
            if len(q) < 10:
                continue
            if not q.endswith("?"):
                q = q.rstrip(".") + "?"
            raw_id = str(entry.get("id") or f"llm_goal_{idx}").strip().lower()
            clean_id = re.sub(r"[^a-z0-9_]+", "_", raw_id).strip("_") or f"llm_goal_{idx}"
            if clean_id in seen_ids:
                clean_id = f"{clean_id}_{idx}"
            seen_ids.add(clean_id)
            items.append({"id": clean_id, "question": q})
            if len(items) >= 4:
                break

    if not items:
        # Deterministic non-privileged fallback derived purely from task_instruction
        items.append(
            {
                "id": "llm_goal_0",
                "question": (
                    f'Task: "{task_instruction.strip()}" Has the main target object been '
                    "placed at its requested destination and released by the robot gripper?"
                ),
            }
        )
    return items


def generate_task_graph_with_qwen(
    scorer: Qwen3VLTernaryScorer,
    task_instruction: str,
    *,
    max_new_tokens: int = 256,
) -> tuple[list[dict[str, str]], str]:
    """Generate visual placement questions from ``task_instruction`` alone via greedy decoding."""
    import torch

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": TASK_GRAPH_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": TASK_GRAPH_USER_TEMPLATE.format(task_instruction=task_instruction.strip()),
                }
            ],
        },
    ]
    prompt = scorer.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tok_inputs = scorer.processor.tokenizer([prompt], return_tensors="pt").to(scorer.device)
    with torch.inference_mode():
        gen_ids = scorer.model.generate(
            **tok_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=scorer.processor.tokenizer.pad_token_id,
        )
    new_tokens = gen_ids[0, tok_inputs["input_ids"].shape[1] :]
    raw_text = scorer.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_llm_task_graph_response(raw_text, task_instruction), raw_text


def score_rollouts_with_llm_task_graphs(
    roots: list[Path],
    output_dir: Path,
    model_dir: Path,
    *,
    tasks_filter: set[str] | None = None,
    decision_stride: int = 2,
    batch_size: int = 16,
    answer_instruction: str = ANSWER_INSTRUCTION,
    existing_graphs_path: Path | None = None,
    scorer: Any = None,
    lora_checkpoint: Path | None = None,
) -> dict:
    """Generate (or load) LLM task graphs per unique ``(task, task_instruction)`` and
    score every matching rollout at ``decision_stride`` with atomic per-rollout resume.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "per_rollout"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = _discover_manifests(roots)
    rows_with_base: list[tuple[Path, dict]] = []
    seen_ids = set()
    input_hashes = {}
    unique_task_instrs: dict[tuple[str, str], str] = {}

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
            if tasks_filter is not None and task not in tasks_filter:
                continue
            seen_ids.add(rid)
            instr = str(row.get("task_instruction") or "").strip()
            unique_task_instrs[(task, instr)] = instr
            rows_with_base.append((base, row))

    if not rows_with_base:
        raise RuntimeError("No matching rollouts found for LLM task-graph scoring")

    if scorer is None:
        scorer = Qwen3VLTernaryScorer(model_dir)

    # Step 1: Load or generate LLM task graphs BEFORE loading any LoRA adapter
    # (so task decomposition uses the pure pretrained instruction-following LLM).
    graphs_path = existing_graphs_path or (output_dir / "llm_task_graphs.json")
    if graphs_path.exists():
        graphs_payload = json.loads(graphs_path.read_text())
        graphs_by_key = graphs_payload["graphs_by_task_instruction"]
        print(f"Loaded {len(graphs_by_key)} cached LLM task graphs from {graphs_path}")
    else:
        graphs_by_key = {}
        raw_outputs = {}
        for (task, instr) in sorted(unique_task_instrs):
            k = f"{task}||{instr}"
            items, raw_txt = generate_task_graph_with_qwen(scorer, instr)
            graphs_by_key[k] = items
            raw_outputs[k] = raw_txt
            print(f"[LLM graph] {task:22s} -> {len(items)} questions: {[x['id'] for x in items]}")
            for x in items:
                print(f"    - {x['id']}: {x['question']}")
        graphs_payload = {
            "schema_version": 1,
            "model_dir": str(model_dir),
            "graphs_by_task_instruction": graphs_by_key,
            "raw_outputs": raw_outputs,
        }
        (output_dir / "llm_task_graphs.json").write_text(json.dumps(graphs_payload, indent=2) + "\n")

    # Optional: load LoRA adapter weights AFTER generating task graphs (for LoRA-scored ablation)
    if lora_checkpoint is not None:
        from .train_eval_vlm_lora_cross_task import inject_lora_adapters

        ckpt = torch.load(str(lora_checkpoint), map_location=scorer.device)
        inject_lora_adapters(
            scorer.model,
            rank=int(ckpt["rank"]),
            alpha=float(ckpt["alpha"]),
            dropout=0.0,
            include_vision=bool(ckpt.get("include_vision", True)),
        )
        sd = ckpt["state_dict"]
        missing, unexpected = scorer.model.load_state_dict(sd, strict=False)
        loaded_lora = sum(1 for k in sd if k not in unexpected)
        scorer.model.eval()
        print(f"Loaded {loaded_lora} LoRA tensors from {lora_checkpoint} for LLM-graph scoring")

    # Step 2: Score every rollout's LLM-generated questions with atomic checkpoints
    episodes_meta = []
    packed_arrays: dict[str, np.ndarray] = {}
    t0 = time.time()
    total_q = 0
    computed = 0
    resumed = 0

    for ep_i, (base, row) in enumerate(rows_with_base):
        rid = row["rollout_id"]
        task = row["task_name"]
        instr = str(row.get("task_instruction") or "").strip()
        q_items = graphs_by_key[f"{task}||{instr}"]
        q_ids = [x["id"] for x in q_items]

        trace = json.loads((base / row["subtask_trace_path"]).read_text())
        inf_records = trace["inference_records"]
        dec_steps = np.array([int(x["environment_step"]) for x in inf_records], dtype=int)
        n_dec = len(dec_steps)
        sampled_pos = sampled_decision_positions(n_dec, decision_stride)

        ckpt_path = ckpt_dir / f"{rid}.json"
        ep_scores = None
        if ckpt_path.exists():
            cand = json.loads(ckpt_path.read_text())
            if (
                cand.get("decision_stride") == int(decision_stride)
                and cand.get("sampled_positions") == sampled_pos.tolist()
                and list(cand.get("predicates", {}).keys()) == q_ids
            ):
                ep_scores = cand
                resumed += 1

        if ep_scores is None:
            f_idx_all = causal_frame_indices(dec_steps, trace["video_frame_environment_steps"])
            f_idx_s = f_idx_all[sampled_pos]
            decoded_views = _decode_video_pil_views(base / row["video_path"], f_idx_s)

            query_items = []
            for entry in q_items:
                q_full = f"{entry['question']}\n{answer_instruction}"
                for fidx in f_idx_s:
                    query_items.append((decoded_views[int(fidx)], q_full))

            ly, ln, lu = scorer.score_batch(query_items, batch_size=batch_size)
            total_q += len(query_items)
            computed += 1

            pred_payload: dict[str, dict[str, list[float]]] = {}
            n_s = len(sampled_pos)
            cur = 0
            for qid in q_ids:
                sl = slice(cur, cur + n_s)
                cur += n_s
                sc = compute_ternary_scores(ly[sl], ln[sl], lu[sl], invert=False)
                pred_payload[qid] = {
                    "sampled_logit_diff": [round(float(x), 5) for x in sc["logit_diff"]],
                    "sampled_prob_yes": [round(float(x), 5) for x in sc["prob_yes"]],
                    "sampled_prob_visible": [round(float(x), 5) for x in sc["prob_visible"]],
                }

            ep_scores = {
                "rollout_id": rid,
                "task": task,
                "task_instruction": instr,
                "num_decisions": n_dec,
                "decision_stride": int(decision_stride),
                "sampled_positions": sampled_pos.tolist(),
                "predicates": pred_payload,
            }
            tmp_ckpt = ckpt_dir / f".{rid}.json.tmp"
            tmp_ckpt.write_text(json.dumps(ep_scores) + "\n")
            tmp_ckpt.replace(ckpt_path)

            if computed in (1, 5) or (ep_i + 1) % 20 == 0 or (ep_i + 1) == len(rows_with_base):
                el = max(1e-6, time.time() - t0)
                print(
                    f"[LLM-graph {ep_i + 1:3d}/{len(rows_with_base)}] {task:21s} {rid[:8]} | "
                    f"q={total_q} ({total_q / el:.1f} q/s) | elapsed={el:.1f}s"
                )

        for qid in q_ids:
            pp = ep_scores["predicates"][qid]
            s_ld = np.asarray(pp["sampled_logit_diff"], dtype=np.float32)
            s_py = np.asarray(pp["sampled_prob_yes"], dtype=np.float32)
            s_pv = np.asarray(pp["sampled_prob_visible"], dtype=np.float32)
            key = f"{rid}__{qid}"
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
                "task_instruction": instr,
                "success": not bool(row["failed"]),
                "num_decisions": n_dec,
                "sampled_positions": sampled_pos.tolist(),
                "predicates": q_ids,
            }
        )

    index_payload = {
        "schema_version": 1,
        "protocol": "llm_generated_task_graph_vlm_scoring",
        "model_dir": str(model_dir),
        "lora_checkpoint": str(lora_checkpoint) if lora_checkpoint is not None else None,
        "graphs_path": str(graphs_path),
        "decision_stride": int(decision_stride),
        "num_episodes": len(episodes_meta),
        "computed_rollouts": computed,
        "resumed_rollouts": resumed,
        "total_queries_computed": total_q,
        "elapsed_sec": round(time.time() - t0, 2),
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
        f"Done LLM-graph scoring: {len(episodes_meta)} rollouts (computed={computed}, "
        f"resumed={resumed}, queries={total_q}) in {time.time() - t0:.1f}s -> {output_dir}"
    )
    return index_payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_QWEN3_VL_DIR))
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--decision-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--existing-graphs", type=Path, default=None)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--binary-prompt", action="store_true")
    args = parser.parse_args(argv)
    ans = "Answer with a single word: Yes or No." if args.binary_prompt else ANSWER_INSTRUCTION
    score_rollouts_with_llm_task_graphs(
        args.run,
        args.output,
        args.model_dir,
        tasks_filter=set(args.tasks) if args.tasks else None,
        decision_stride=args.decision_stride,
        batch_size=args.batch_size,
        answer_instruction=ans,
        existing_graphs_path=args.existing_graphs,
        lora_checkpoint=args.lora_checkpoint,
    )


if __name__ == "__main__":
    main()
