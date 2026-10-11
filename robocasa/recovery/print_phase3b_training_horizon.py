"""Print a running or final Phase 3B continuation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocasa.recovery.counterfactual_branch import load_branch_records
from robocasa.recovery.phase3b_training_horizon import read_jsonl


def _compact_error(value, limit=600):
    value = str(value or "")
    if len(value) <= limit:
        return value
    return (
        value[:limit]
        + f"... [truncated {len(value) - limit} characters; inspect errors.jsonl]"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_dir
    registration_path = root / "registration.json"
    if not registration_path.is_file():
        print("PHASE 3B TRAINING-HORIZON CONTINUATION")
        print("status: initializing_registration")
        print("registration is not frozen yet; source hashes are still being built")
        return
    registration = json.loads(registration_path.read_text())
    status = (
        json.loads((root / "status.json").read_text())
        if (root / "status.json").is_file()
        else {"status": "registered"}
    )
    records = load_branch_records(root / "branch_records.jsonl")
    references = load_branch_records(
        root / "prefix_references" / "branch_records.jsonl"
    )
    errors = read_jsonl(root / "errors.jsonl")
    print("PHASE 3B TRAINING-HORIZON CONTINUATION")
    print("status:", status["status"])
    print("scope:", registration["scope"])
    print(
        "parents:",
        len({row["parent_id"] for row in records}),
        "/",
        registration["expected_parents"],
    )
    print(
        "snapshots:",
        len({row["snapshot_id"] for row in records}),
        "/",
        registration["expected_snapshots"],
    )
    print("branches:", len(records), "/", registration["expected_branches"])
    print(
        "prefix references:",
        len(references),
        "/",
        registration["expected_branches"],
    )
    print(
        "first-64 exact:",
        sum(bool(row.get("first64_exact")) for row in records),
        "/",
        len(records),
    )
    print("errors:", len(errors))
    if status["status"] == "failed":
        print(
            "failure:",
            status.get("error_type"),
            _compact_error(status.get("error")),
        )
        if errors and errors[-1].get("first64_audit"):
            audit = errors[-1]["first64_audit"]
            legacy = audit.get("legacy_phase2_qualification") or {}
            print("legacy Phase 2 prefix qualified:", legacy.get("qualified"))
            print("legacy qualification mode:", legacy.get("mode"))
            if audit.get("reference_to_continuation") is not None:
                audit_scope = "reference-to-continuation"
                reference_audit = audit["reference_to_continuation"]
            else:
                audit_scope = "legacy-Phase-2-to-reference"
                reference_audit = audit.get("legacy_phase2") or audit
            print("displayed first-64 audit:", audit_scope)
            if legacy.get("portable_channels"):
                print("legacy portable channels:")
                for name, exact in sorted(legacy["portable_channels"].items()):
                    print(f"  {'PASS' if exact else 'FAIL':4s}  {name}")
            print(
                "legacy terminal camera observation qualified:",
                legacy.get("terminal_camera_observation_qualified"),
            )
            if legacy.get("observation_component_mismatches"):
                print(
                    "legacy observation component mismatches:",
                    legacy["observation_component_mismatches"],
                )
            print(
                "legacy causal environment mismatch indices:",
                legacy.get("causal_environment_mismatch_indices"),
            )
            print(
                "legacy first-transition model XML compatible:",
                legacy.get("first_causal_transition_model_xml_compatible"),
            )
            print(
                "legacy first-transition model XML mode:",
                legacy.get("first_causal_transition_model_xml_compatibility_mode"),
            )
            failed_channels = [
                name
                for name, exact in reference_audit.get("channels", {}).items()
                if not exact
            ]
            print("failed first-64 channels:", " ".join(failed_channels))
            for name, indices in sorted(
                reference_audit.get("sequence_mismatch_indices", {}).items()
            ):
                print(f"  {name} mismatch indices:", indices)
            print(
                "first causal transition structure exact:",
                reference_audit.get("first_causal_transition_structure_exact"),
            )
            mismatches = reference_audit.get(
                "first_causal_transition_structure_mismatches", []
            )
            for mismatch in mismatches[:10]:
                print("  transition mismatch:", json.dumps(mismatch, sort_keys=True))
            if len(mismatches) > 10:
                print("  additional transition mismatches:", len(mismatches) - 10)
            if errors[-1].get("diagnostic_payload_path"):
                print(
                    "diagnostic payload:",
                    root / errors[-1]["diagnostic_payload_path"],
                )
            if errors[-1].get("prefix_reference_payload_path"):
                print(
                    "prefix reference payload:",
                    root / errors[-1]["prefix_reference_payload_path"],
                )
    if not (root / "analysis.json").is_file():
        print("Engineering analysis is not frozen yet.")
        return
    analysis = json.loads((root / "analysis.json").read_text())
    model_xml = analysis.get("environment_model_xml_restore") or {}
    if model_xml:
        print("\nMODEL XML RESTORE")
        print(
            "audited complete restores:",
            model_xml["audited_complete_restores"],
            "/",
            model_xml["expected_complete_restores"],
        )
        print("compatibility modes:", model_xml["compatibility_modes"])
        print(
            "allowed OBJ MIME normalizations:",
            model_xml.get(
                "allowed_mesh_content_type_normalizations",
                model_xml.get("allowed_mesh_content_type_omissions", 0),
            ),
        )
        print(
            "  omissions:",
            model_xml.get("allowed_mesh_content_type_omissions", 0),
            "insertions:",
            model_xml.get("allowed_mesh_content_type_insertions", 0),
        )
    print("\nENGINEERING GATES")
    for name, passed in analysis["engineering_gates"].items():
        print(f"{'PASS' if passed else 'FAIL':4s}  {name}")
    print(
        "\nPHASE 3B ENGINEERING VALIDITY:",
        "PASS" if analysis["engineering_all_pass"] else "FAIL",
    )
    scientific_path = root / "scientific_analysis.json"
    if scientific_path.is_file():
        scientific = json.loads(scientific_path.read_text())
        print("routing decision:", scientific["routing_decision"])


if __name__ == "__main__":
    main()
