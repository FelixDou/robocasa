"""Extract strictly causal camera, proprioception, action, and XR-1 features
aligned to every XR-1 policy decision across synchronized recovery pilot runs.

Designed for offline cluster execution using the locally cached
``google/paligemma-3b-pt-224`` SigLIP-So400M/14-224 vision tower (no network
access required). Saves a single compact ``perception_features.npz`` plus
``perception_index.json`` for fast CPU/GPU probing and end-to-end camera-alarm
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np

from . import analyze_recovery_pilot6 as pilot6


DEFAULT_PALIGEMMA_SNAPSHOT = (
    "/gs/bs/tga-shinoda/felid/hf_home/hub/models--google--paligemma-3b-pt-224/"
    "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
)


def causal_frame_indices(decision_steps: np.ndarray, video_frame_steps: list[int]) -> np.ndarray:
    """Map each policy decision step ``t`` to the latest video frame index with step <= t.

    Strictly causal for all ``t > 0``: never reads any frame rendered after
    decision ``t``. For ``t == 0`` (before the first post-action frame at step 1),
    returns index 0 (step 1) as the reset-state visual proxy.
    """
    if not video_frame_steps:
        raise ValueError("video_frame_environment_steps is empty")
    fsteps = np.asarray(video_frame_steps, dtype=int)
    indices = np.empty(len(decision_steps), dtype=int)
    for i, t in enumerate(decision_steps):
        pos = int(np.searchsorted(fsteps, int(t), side="right")) - 1
        indices[i] = max(0, pos)
    return indices


def split_obs_grid_3cam(bgr_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 768x256 ``obs_grid`` BGR frame into RGB (left, right, wrist) crops."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    if w % 3 != 0:
        raise ValueError(f"Expected 3-view horizontal obs_grid, got shape {rgb.shape}")
    cw = w // 3
    return rgb[:, :cw], rgb[:, cw : 2 * cw], rgb[:, 2 * cw :]


def preprocess_siglip_rgb(img_rgb: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Resize RGB uint8 image to ``image_size`` and normalize to [-1, 1] CHW float32."""
    resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) * (1.0 / 255.0)
    arr = (arr - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))


def summarize_executed_actions(actions_npz: dict[str, np.ndarray], decision_steps: np.ndarray, replan_steps: int = 16) -> np.ndarray:
    """Build strictly causal executed-action summary over ``[max(0, t - replan_steps), t)``.

    Returns shape ``(num_decisions, 18)`` float32:
    gripper (mean, min, max, last, delta), EEF pos (mean3, norm_delta),
    EEF rot (mean3, norm_delta), base motion (mean4, norm_mean).
    """
    grip = np.asarray(actions_npz["action_gripper_close"], dtype=np.float32).reshape(-1)
    eef_pos = np.asarray(actions_npz["action_end_effector_position"], dtype=np.float32)
    eef_rot = np.asarray(actions_npz["action_end_effector_rotation"], dtype=np.float32)
    base = np.asarray(actions_npz["action_base_motion"], dtype=np.float32)

    out = np.zeros((len(decision_steps), 18), dtype=np.float32)
    for i, t in enumerate(decision_steps):
        end = int(t)
        if end <= 0:
            continue
        start = max(0, end - replan_steps)
        g = grip[start:end]
        ep = eef_pos[start:end]
        er = eef_rot[start:end]
        bm = base[start:end]
        if len(g) == 0:
            continue
        out[i, 0] = float(g.mean())
        out[i, 1] = float(g.min())
        out[i, 2] = float(g.max())
        out[i, 3] = float(g[-1])
        out[i, 4] = float(g[-1] - g[0])
        out[i, 5:8] = ep.mean(axis=0)
        out[i, 8] = float(np.linalg.norm(ep, axis=1).mean())
        out[i, 9:12] = er.mean(axis=0)
        out[i, 12] = float(np.linalg.norm(er, axis=1).mean())
        out[i, 13:17] = bm.mean(axis=0)
        out[i, 17] = float(np.linalg.norm(bm, axis=1).mean())
    return out


def summarize_action_chunks(chunks: np.ndarray) -> np.ndarray:
    """Summarize XR-1 predicted action chunks ``(D, H, A)`` into ``(D, 3 * A)`` float32."""
    c = np.asarray(chunks, dtype=np.float32)
    return np.concatenate([c.mean(axis=1), c.std(axis=1), c[:, -1, :] - c[:, 0, :]], axis=1)


def load_siglip_vision_tower(snapshot_dir: Path, device):
    """Load ONLY the SigLIP vision tower from a cached PaliGemma snapshot.

    Reads ``vision_tower.vision_model.*`` directly from the safetensors shards
    to avoid loading the 2B Gemma language model into VRAM. PaliGemma omits
    SigLIP's optional attention-pooling ``head``, so we always pool the trained
    post-layernorm patch tokens (``last_hidden_state.mean(dim=1)``).
    """
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel

    cfg_path = snapshot_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    vision_cfg = SiglipVisionConfig.from_dict(cfg["vision_config"])
    image_size = int(vision_cfg.image_size)

    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        from safetensors.torch import load_file

        weight_map = json.loads(index_path.read_text())["weight_map"]
        prefix = "vision_tower."
        by_shard: dict[str, list[str]] = {}
        for k, shard in weight_map.items():
            if k.startswith(prefix):
                by_shard.setdefault(shard, []).append(k)
        state_dict = {}
        for shard, keys in by_shard.items():
            shard_tensors = load_file(str(snapshot_dir / shard), device="cpu")
            for k in keys:
                state_dict[k[len(prefix) :]] = shard_tensors[k]
        model = SiglipVisionModel(vision_cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing_non_head = [k for k in missing if ".head." not in k]
        if missing_non_head:
            raise RuntimeError(
                f"Missing SigLIP vision weights: {missing_non_head[:5]} (total {len(missing_non_head)})"
            )
        if missing:
            print(f"Note: {len(missing)} optional SigLIP head keys absent (expected for PaliGemma); using patch mean pooling")
        if unexpected:
            print(f"Note: ignored {len(unexpected)} unexpected keys when loading SigLIP tower")
    else:
        from transformers import AutoModel

        full = AutoModel.from_pretrained(str(snapshot_dir), torch_dtype=torch.bfloat16, local_files_only=True)
        model = full.vision_tower
        del full

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    return model, image_size, dtype


def encode_video_decisions(
    video_path: Path,
    frame_indices: np.ndarray,
    model,
    device,
    dtype,
    image_size: int = 224,
    batch_size: int = 128,
) -> np.ndarray:
    """Decode only the needed frames from ``video_path`` and return ``(D, 3, H_vis)`` float16."""
    import torch

    needed = set(int(x) for x in frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    decoded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in needed:
                l_rgb, r_rgb, w_rgb = split_obs_grid_3cam(bgr)
                decoded[idx] = (
                    preprocess_siglip_rgb(l_rgb, image_size),
                    preprocess_siglip_rgb(r_rgb, image_size),
                    preprocess_siglip_rgb(w_rgb, image_size),
                )
            idx += 1
    finally:
        cap.release()

    missing = needed - set(decoded)
    if missing:
        raise RuntimeError(f"Video {video_path} missing requested frames: {sorted(missing)[:5]}")

    unique_sorted = sorted(needed)
    flat_imgs = []
    for fidx in unique_sorted:
        flat_imgs.extend(decoded[fidx])
    flat_arr = np.stack(flat_imgs, axis=0)  # (3 * U, 3, H, W)

    emb_chunks = []
    with torch.no_grad():
        for start in range(0, len(flat_arr), batch_size):
            batch = torch.from_numpy(flat_arr[start : start + batch_size]).to(device=device, dtype=dtype)
            out = model(pixel_values=batch)
            vec = out.last_hidden_state.mean(dim=1)
            emb_chunks.append(vec.float().cpu().numpy().astype(np.float16))
    flat_emb = np.concatenate(emb_chunks, axis=0)
    per_unique = flat_emb.reshape(len(unique_sorted), 3, -1)
    u_map = {fidx: i for i, fidx in enumerate(unique_sorted)}
    return np.stack([per_unique[u_map[int(fidx)]] for fidx in frame_indices], axis=0)


def extract_dataset(
    roots: list[Path],
    output_dir: Path,
    vision_snapshot: Path | None,
    *,
    batch_size: int = 128,
    skip_vision: bool = False,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    manifest_paths = []
    for r in roots:
        if (r / "manifest.jsonl").exists():
            manifest_paths.append(r / "manifest.jsonl")
        else:
            manifest_paths.extend(sorted(r.glob("shard*/manifest.jsonl")))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifest.jsonl under {[str(r) for r in roots]}")

    model = None
    image_size = 224
    dtype = None
    device = None
    if not skip_vision:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if vision_snapshot is None or not vision_snapshot.exists():
            raise FileNotFoundError(f"Vision snapshot not found: {vision_snapshot}")
        t0 = time.time()
        model, image_size, dtype = load_siglip_vision_tower(vision_snapshot, device)
        print(f"Loaded SigLIP vision tower on {device} ({dtype}) in {time.time() - t0:.1f}s")

    episodes_meta = []
    siglip_list = []
    dit_last_list = []
    dit_mean_list = []
    state_list = []
    chunk_list = []
    exec_act_list = []
    label_rows = []
    seen_ids = set()
    input_hashes = {}

    t_start = time.time()
    offset = 0
    for mp in sorted(manifest_paths):
        base = mp.parent
        input_hashes[str(mp)] = hashlib.sha256(mp.read_bytes()).hexdigest()
        rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
        for r_idx, row in enumerate(rows):
            rid = row["rollout_id"]
            if rid in seen_ids:
                raise ValueError(f"Duplicate rollout_id: {rid}")
            seen_ids.add(rid)
            task = row["task_name"]
            trace = json.loads((base / row["subtask_trace_path"]).read_text())
            rollout_npz = np.load(base / row["tensor_path"])
            actions_npz = np.load(base / row["action_path"])

            inf_records = trace["inference_records"]
            dec_steps = np.array([int(x["environment_step"]) for x in inf_records], dtype=int)
            n_dec = len(dec_steps)
            n_steps = int(row["num_environment_steps"])
            replan = int(row.get("replan_steps") or 16)

            # Verify alignment with saved SAFE tensor
            stored_steps = np.asarray(rollout_npz["inference_environment_steps"], dtype=int)
            if not np.array_equal(dec_steps, stored_steps):
                raise ValueError(f"{rid}: inference_environment_steps mismatch between trace and tensor")

            feats = rollout_npz["features"]  # (D, 5, 16, 1024)
            dit_last = np.asarray(feats[:, -1, -1, :], dtype=np.float16)
            dit_mean = np.asarray(feats[:, -1, :, :].mean(axis=1), dtype=np.float16)
            state_hist = np.asarray(rollout_npz["auxiliary__observation_state_history"], dtype=np.float32)
            chunk_sum = summarize_action_chunks(rollout_npz["policy_action_chunks"])
            exec_sum = summarize_executed_actions(actions_npz, dec_steps, replan_steps=replan)

            dit_last_list.append(dit_last)
            dit_mean_list.append(dit_mean)
            state_list.append(state_hist)
            chunk_list.append(chunk_sum)
            exec_act_list.append(exec_sum)

            if not skip_vision:
                f_idx = causal_frame_indices(dec_steps, trace["video_frame_environment_steps"])
                vis = encode_video_decisions(
                    base / row["video_path"], f_idx, model, device, dtype,
                    image_size=image_size, batch_size=batch_size,
                )
                siglip_list.append(vis)

            # Ground-truth relation truth at every decision step t (strictly at step t).
            hist = trace["recovery_observations"]
            pred_names = sorted(hist[0]["values"].keys())
            series = {p: pilot6.expand_recovery_series(hist, n_steps, p) for p in pred_names}
            pred_at_dec = {p: [bool(series[p][int(t)]) for t in dec_steps] for p in pred_names}

            episodes_meta.append(
                {
                    "rollout_id": rid,
                    "task": task,
                    "seed": int(row["environment_seed"]),
                    "success": not bool(row["failed"]),
                    "horizon": int(row["rollout_horizon"]),
                    "num_steps": n_steps,
                    "replan_steps": replan,
                    "start_row": offset,
                    "end_row": offset + n_dec,
                    "decision_steps": dec_steps.tolist(),
                    "decision_stages": [str(x["subtask_id"]) for x in inf_records],
                    "predicates_at_decisions": pred_at_dec,
                    "segments": trace.get("segments", []),
                    "recovery_observations": hist,
                }
            )
            offset += n_dec
            if (len(episodes_meta)) % 20 == 0:
                print(f"Processed {len(episodes_meta)} rollouts ({offset} decisions) in {time.time() - t_start:.1f}s")

    arrays = {
        "dit_last": np.concatenate(dit_last_list, axis=0),
        "dit_mean": np.concatenate(dit_mean_list, axis=0),
        "state_history": np.concatenate(state_list, axis=0).astype(np.float32),
        "action_chunks": np.concatenate(chunk_list, axis=0).astype(np.float32),
        "executed_actions": np.concatenate(exec_act_list, axis=0).astype(np.float32),
    }
    if siglip_list:
        arrays["siglip_3cam"] = np.concatenate(siglip_list, axis=0)  # (N_dec, 3, 1152) fp16

    index_payload = {
        "schema_version": 1,
        "vision_snapshot": str(vision_snapshot) if not skip_vision else None,
        "num_episodes": len(episodes_meta),
        "num_decisions": offset,
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "inputs_sha256": input_hashes,
        "episodes": episodes_meta,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        np.savez(staged / "perception_features.npz", **arrays)
        (staged / "perception_index.json").write_text(json.dumps(index_payload) + "\n")
        staged.rename(output_dir)

    print(
        f"Done: {len(episodes_meta)} rollouts, {offset} decisions in {time.time() - t_start:.1f}s -> {output_dir}"
    )
    print("Shapes:", index_payload["array_shapes"])
    return index_payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-snapshot", type=Path, default=Path(DEFAULT_PALIGEMMA_SNAPSHOT))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-vision", action="store_true")
    args = parser.parse_args(argv)
    extract_dataset(
        args.run,
        args.output,
        None if args.skip_vision else args.vision_snapshot,
        batch_size=args.batch_size,
        skip_vision=args.skip_vision,
    )


if __name__ == "__main__":
    main()
