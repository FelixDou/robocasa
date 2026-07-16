"""Compute LingBot interface normalization from RoboCasa LeRobot parquet data.

This utility performs interface calibration only. It reads the low-dimensional
``observation.state`` and ``action`` columns and never reads images, language,
rewards, success labels, or task annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FEATURE_SLICES = {
    # RoboCasa LeRobot state ordering is already the 16-D PandaOmron ordering.
    "observation.state.end.position": ("state", slice(7, 14)),
    "observation.state.effector.position": ("state", slice(14, 16)),
    "observation.state.base.position": ("state", slice(0, 3)),
    # RoboCasa LeRobot action ordering comes from PandaOmron_modality.json:
    # base motion [0:4], control mode [4], EEF pos [5:8], EEF rot [8:11],
    # and gripper [11]. The discrete control-mode column is intentionally omitted.
    "action.end.position": ("action", slice(5, 11)),
    "action.effector.position": ("action", slice(11, 12)),
    "action.base.position": ("action", slice(0, 3)),
    "action.waist.position": ("action", slice(3, 4)),
}


def mapped_features(state: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 16:
        raise ValueError(f"Expected RoboCasa states shaped (N, 16), got {state.shape}")
    if action.ndim != 2 or action.shape[1] != 12:
        raise ValueError(f"Expected RoboCasa actions shaped (N, 12), got {action.shape}")
    if state.shape[0] != action.shape[0]:
        raise ValueError(
            f"State/action row counts differ: {state.shape[0]} != {action.shape[0]}"
        )

    arrays = {"state": state, "action": action}
    return {
        key: np.ascontiguousarray(arrays[source][:, feature_slice])
        for key, (source, feature_slice) in FEATURE_SLICES.items()
    }


def feature_statistics(value: np.ndarray) -> dict[str, list[float]]:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2-D feature array, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Feature array contains NaN or infinite values")
    return {
        "mean": value.mean(axis=0).tolist(),
        "std": value.std(axis=0).tolist(),
        "min": value.min(axis=0).tolist(),
        "max": value.max(axis=0).tolist(),
        "q01": np.quantile(value, 0.01, axis=0).tolist(),
        "q99": np.quantile(value, 0.99, axis=0).tolist(),
    }


def find_parquet_files(dataset_roots: list[Path]) -> list[Path]:
    files = set()
    for root in dataset_roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        if root.is_file():
            if root.suffix != ".parquet":
                raise ValueError(f"Dataset file is not parquet: {root}")
            files.add(root)
            continue
        files.update(root.glob("data/*/*.parquet"))
        files.update(root.glob("lerobot/data/*/*.parquet"))
        files.update(root.glob("**/lerobot/data/*/*.parquet"))
    return sorted(files)


def read_low_dimensional_columns(
    parquet_files: list[Path], batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read RoboCasa parquet files") from exc

    state_chunks = []
    action_chunks = []
    total = 0
    for index, path in enumerate(parquet_files, start=1):
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        missing = {"observation.state", "action"} - columns
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["observation.state", "action"],
        ):
            state = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
            action = np.asarray(batch.column(1).to_pylist(), dtype=np.float32)
            # Validate shapes before accumulating potentially many files.
            mapped_features(state, action)
            state_chunks.append(state)
            action_chunks.append(action)
            total += state.shape[0]
        print(f"[{index}/{len(parquet_files)}] rows={total} {path}")

    if not state_chunks:
        raise RuntimeError("No rows were read from the parquet inputs")
    return np.concatenate(state_chunks), np.concatenate(action_chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute data-only LingBot normalization for RoboCasa."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        required=True,
        help="LeRobot directory, its parent, or a parquet file. Repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    parquet_files = find_parquet_files(args.dataset_root)
    if not parquet_files:
        raise FileNotFoundError("No LeRobot parquet files found under dataset roots")

    print(f"parquet_files={len(parquet_files)}")
    state, action = read_low_dimensional_columns(parquet_files, args.batch_size)
    features = mapped_features(state, action)
    output = {
        "norm_stats": {
            key: feature_statistics(value) for key, value in features.items()
        },
        "count": int(state.shape[0]),
        "calibration": {
            "type": "data_statistics_only",
            "normalization": "meanstd",
            "columns_read": ["observation.state", "action"],
            "excluded_action_index": 4,
            "excluded_modalities": [
                "images",
                "language",
                "rewards",
                "success_labels",
                "task_annotations",
            ],
            "parquet_file_count": len(parquet_files),
            "dataset_roots": [str(path.expanduser().resolve()) for path in args.dataset_root],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"rows={state.shape[0]}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
