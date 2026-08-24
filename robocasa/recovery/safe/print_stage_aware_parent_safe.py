"""Print compact development or prospective stage-aware SAFE results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _number(value):
    return "n/a" if value is None else f"{float(value):.4f}"


def print_development(root, analysis):
    bundle = json.loads((root / "runtime_bundle.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    print("FULL-PARENT STAGE-AWARE SAFE DEVELOPMENT: COMPLETE")
    print("opened outer scored:", analysis["opened_outer_scored"])
    print("development parents:", sum(protocol["development_allocation"]["counts"].values()))
    print("locked opened outer parents:", protocol["locked_opened_outer_parents"])
    print("selected stages:", len(protocol["selected_stages"]))
    print("available arms:", " ".join(protocol["available_arms"]))
    if protocol["unavailable_arms"]:
        print("unavailable arms:", protocol["unavailable_arms"])
    print("primary detector:", bundle["primary_detector"])
    print("\nSELECTION (prefix-1/2 task-stage macro ROC-AUC)")
    for detector, value in sorted(bundle["selection_values"].items()):
        print(f"{detector:16s} {_number(value)}")
    print("\nSUCCESS-ONLY CALIBRATION")
    for detector in bundle["detectors"]:
        threshold = bundle["thresholds"][detector]
        metrics = bundle["calibration_metrics"][detector]
        parent = metrics["parent_level"]
        pooled = metrics["pooled"]
        print(
            f"{detector:16s} threshold={threshold['threshold']:.6f} "
            f"parent_FPR={_number(parent['successful_parent_fpr'])} "
            f"failed-stage_TPR={_number(parent['failed_stage_tpr'])} "
            f"stage_bal_acc={_number(pooled['balanced_accuracy'])}"
        )
    print("\nruntime bundle:", root / "runtime_bundle.json")


def print_prospective(root, analysis):
    if analysis.get("prospective_claim", True):
        print("FULL-PARENT STAGE-AWARE SAFE PROSPECTIVE TEST: COMPLETE")
    else:
        print("FULL-PARENT STAGE-AWARE SAFE RETROSPECTIVE OUTER CHECK: COMPLETE")
    print("evaluation scope:", analysis.get("evaluation_scope", "prospective"))
    print("prospective claim:", analysis.get("prospective_claim", True))
    print("parents:", analysis["parents"])
    print("tasks:", len(analysis["tasks"]))
    print("stage events:", analysis["stage_events"])
    print(
        "unobserved failed-stage parents:",
        analysis.get("unobserved_failed_stage_parents", 0),
    )
    print("thresholds updated on test:", analysis["thresholds_updated_on_test"])
    print("development IDs disjoint:", analysis["development_ids_disjoint"])
    print("primary detector:", analysis["primary_detector"])
    ranking = analysis.get("ranking_confirmation")
    if ranking is not None:
        print("\nFIXED-PREFIX TASK-STAGE MACRO ROC-AUC")
        print("primary prefixes:", ranking["primary_prefixes"])
        for detector, value in sorted(ranking["primary_values"].items()):
            print(f"{detector:16s} {_number(value)}")
        print(
            "primary minus terminal:",
            _number(ranking.get("primary_minus_terminal")),
        )
        print(
            "primary minus time_only:",
            _number(ranking.get("primary_minus_time_only")),
        )
        print("tasks primary beats terminal:", ranking["tasks_primary_beats_terminal"])
        print("tasks primary beats time_only:", ranking["tasks_primary_beats_time_only"])
        print("\nPAIRED FIXED-PREFIX PRIMARY MINUS BASELINE")
        for baseline, result in sorted(
            ranking.get("paired_task_parent_bootstrap", {}).items()
        ):
            print(
                f"{baseline:16s} delta={result['point']:+.4f} "
                f"95%CI=[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}] "
                f"estimable={result['replicates_estimable']}"
            )
        print("\nRANKING CONFIRMATION CRITERIA")
        for key, value in ranking["criteria"].items():
            print(f"{key}: {value}")
    print("\nMATCHED-FPR PARENT RESULTS")
    for detector, metrics in sorted(analysis["detectors"].items()):
        parent = metrics["parent_level"]
        pooled = metrics["pooled"]
        print(
            f"{detector:16s} parent_FPR={_number(parent['successful_parent_fpr'])} "
            f"failed-stage_TPR={_number(parent['failed_stage_tpr'])} "
            f"adjusted={_number(parent['missed_failure_adjusted_detection_fraction'])} "
            f"stage_bal_acc={_number(pooled['balanced_accuracy'])}"
        )
    print("\nPAIRED ADJUSTED DETECTION FRACTION MINUS TIME")
    for detector, result in sorted(analysis["paired_bootstrap"].items()):
        print(
            f"{detector:16s} delta={result['point']:+.4f} "
            f"95%CI=[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]"
        )
    print("\nPREREGISTERED SUCCESS CRITERIA")
    for key, value in analysis["preregistered_success_criteria"].items():
        print(f"{key}: {value}")
    print("\nartifacts:", root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.run_dir).resolve()
    analysis = json.loads((root / "analysis.json").read_text())
    protocol = analysis.get("protocol")
    if protocol in (
        "prospective_full_parent_stage_aware_safe",
        "retrospective_locked_outer_full_parent_stage_aware_safe",
    ):
        print_prospective(root, analysis)
    elif (root / "runtime_bundle.json").is_file():
        print_development(root, analysis)
    else:
        raise SystemExit("Run directory is neither a complete development nor prospective result")


if __name__ == "__main__":
    main()
