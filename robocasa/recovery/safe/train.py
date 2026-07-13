"""Train SAFE-MLP or SAFE-LSTM from binary rollout outcomes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from .dataset import aggregate_features, load_manifest, load_rollout, pad_sequences
from .models import (
    ModelConfig,
    make_model,
    masked_binary_cross_entropy,
    require_torch,
    save_checkpoint,
    torch,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _load_sequences(dataset_dir, records, aggregation):
    return {
        record.rollout_id: aggregate_features(
            load_rollout(dataset_dir, record)[0], aggregation
        )
        for record in records
    }


def _batch(ids, record_by_id, sequences, device):
    features, mask = pad_sequences([sequences[x] for x in ids])
    lengths = mask.sum(axis=1)
    labels = np.asarray([int(record_by_id[x].failed) for x in ids], dtype=np.float32)
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(lengths).to(device),
        torch.from_numpy(labels).to(device),
    )


def train_model(args):
    require_torch()
    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    records = load_manifest(dataset_dir)
    record_by_id = {record.rollout_id: record for record in records}
    splits = json.loads(Path(args.splits).read_text())
    sequences = _load_sequences(dataset_dir, records, args.aggregation)
    input_dim = next(iter(sequences.values())).shape[-1]
    config = ModelConfig(
        args.model,
        input_dim,
        args.hidden_dim,
        args.num_layers if args.num_layers is not None else (2 if args.model == "mlp" else 1),
        args.dropout,
        args.cumulative_mlp,
    )
    model = make_model(config).to(args.device)
    train_ids = splits["train"]
    if not train_ids:
        raise ValueError("Training split is empty")
    _, _, _, all_labels = _batch(train_ids, record_by_id, sequences, args.device)
    failures = float(all_labels.sum().item())
    successes = len(train_ids) - failures
    class_weights = (
        torch.tensor(len(train_ids) / max(1.0, 2 * successes), device=args.device),
        torch.tensor(len(train_ids) / max(1.0, 2 * failures), device=args.device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        model.train()
        ids = list(train_ids)
        random.shuffle(ids)
        epoch_losses = []
        for start in range(0, len(ids), args.batch_size):
            batch_ids = ids[start : start + args.batch_size]
            features, mask, lengths, labels = _batch(
                batch_ids, record_by_id, sequences, args.device
            )
            optimizer.zero_grad()
            scores = model(features, lengths)
            loss = masked_binary_cross_entropy(scores, labels, mask, class_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("SAFE training loss became non-finite")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)))
    checkpoint = save_checkpoint(
        args.output_checkpoint,
        model,
        config,
        {"aggregation": args.aggregation, "history": history, "seed": args.seed},
    )
    model.eval()
    id_to_split = {
        rollout_id: split
        for split in ("train", "calibration", "seen_test", "unseen_test")
        for rollout_id in splits.get(split, [])
    }
    score_records = []
    with torch.no_grad():
        for record in records:
            sequence = torch.from_numpy(sequences[record.rollout_id][None]).to(args.device)
            lengths = torch.tensor([sequence.shape[1]], device=args.device)
            scores = model(sequence, lengths)[0].detach().cpu().numpy().tolist()
            score_records.append(
                {
                    "rollout_id": record.rollout_id,
                    "task_name": record.task_name,
                    "failed": record.failed,
                    "split": id_to_split[record.rollout_id],
                    "num_env_steps": record.num_env_steps,
                    "num_inferences": record.valid_sequence_length,
                    "environment_seed": record.environment_seed,
                    "checkpoint": record.checkpoint,
                    "policy_id": record.policy_id,
                    "feature_shape": record.feature_shape,
                    "scores": scores,
                }
            )
    scores_path = Path(args.output_scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(score_records, indent=2, sort_keys=True) + "\n")
    return checkpoint, scores_path, history


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--model", choices=["mlp", "lstm"], required=True)
    parser.add_argument("--aggregation", default="mean")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Defaults to official SAFE values: 2 for MLP, 1 for LSTM.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cumulative-mlp", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-scores", required=True)
    return parser


def main(argv=None):
    train_model(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
