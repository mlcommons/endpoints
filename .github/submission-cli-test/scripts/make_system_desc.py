# SPDX-FileCopyrightText: Copyright (c) 2026 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Author the ``system_desc.json`` that ``runs create`` requires (rules §8.2).

``system_desc.json`` is deliberately *not* an endpoints artifact — the submitter
writes it and drops it into the run folder (see the submission CLI's
``docs/endpoints-cli/reference/run-folder-layout.md``). For the sample sweep
there is no real system to describe, so this fills the §8.2 schema with
clearly-marked simulated values and pulls the few fields that must agree with
the run itself out of the run's own artifacts.

Fields derived from the run rather than hardcoded:

* ``model_id`` / ``dataset_name`` — read from the run's ``config.yaml``.
* ``input_token_average`` / ``output_token_average`` — read from
  ``performance/result_summary.json``, so they describe the actual traffic.
* ``measured_accuracy_score`` — read from ``accuracy/accuracy_results.json``.

    python .github/submission-cli-test/scripts/make_system_desc.py --run-dir results/point_c16 --c-max 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

#: Marks every field that describes hardware which does not exist. Anything a
#: reviewer might mistake for a real disclosure carries this string.
_SIMULATED = "SIMULATED - dataset replay server, no physical system under test"


def _mean(stat_block: Any) -> float | None:
    """Pull the mean out of a result_summary stat block, tolerating absence.

    Token-length blocks come back as an empty list when no tokenizer was
    attached, so this must cope with more than just a missing key.
    """
    if isinstance(stat_block, dict):
        value = stat_block.get("avg")
        if isinstance(value, int | float):
            return float(value)
    return None


def _accuracy_score(run_dir: Path) -> float | None:
    """Return the first dataset's accuracy score, or None if no accuracy phase ran."""
    path = run_dir / "accuracy" / "accuracy_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("accuracy_scores") or []
    for entry in scores:
        value = entry.get("score")
        if isinstance(value, int | float):
            return float(value)
    return None


def build_system_desc(run_dir: Path, c_max: int, system_name: str) -> dict[str, Any]:
    """Assemble the §8.2 document for a single measurement point."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    summary = json.loads(
        (run_dir / "performance" / "result_summary.json").read_text(encoding="utf-8")
    )

    model_id = config.get("model_params", {}).get("name", "unknown")
    datasets = config.get("datasets") or [{}]
    dataset_name = datasets[0].get("name", "unknown")

    isl_avg = _mean(summary.get("input_sequence_lengths"))
    osl_avg = _mean(summary.get("output_sequence_lengths"))
    accuracy = _accuracy_score(run_dir)

    return {
        # Org / submission metadata
        "submitter_org_names": "MLCommons",
        "submitter_contact": "endpoints-wg@mlcommons.org",
        # System metadata
        "system_name": system_name,
        "shortened_system_name": system_name[:20],
        "system_category": "datacenter",
        "system_availability_status": "RDI",
        "max_supported_concurrency": c_max,
        "system_size": "1x node, 0x accelerators (simulated)",
        "system_node_ensemble_count": 1,
        "system_node_ensemble_total": 1,
        "serving_framework": "inference_endpoint.testing.dataset_replay_server",
        "node_types": [
            {
                "system_node_ensemble_id": 0,
                "number_of_nodes": 1,
                "host_processor_model_name": _SIMULATED,
                "host_processors_per_node": 1,
                "host_processor_vcpu_count": 4,
                "host_memory_capacity": _SIMULATED,
                "host_memory_configuration": _SIMULATED,
                "accelerator_model_name": _SIMULATED,
                "accelerators_per_node": 0,
                "accelerator_memory_capacity": _SIMULATED,
                "accelerator_memory_type": _SIMULATED,
                "accelerator_interconnect": _SIMULATED,
                "accelerator_host_interconnect": _SIMULATED,
                "host_network_card_count": _SIMULATED,
                "host_networking": _SIMULATED,
                "host_storage_capacity": _SIMULATED,
                "host_storage_type": _SIMULATED,
                "other_hardware": _SIMULATED,
                "cooling": _SIMULATED,
                "hw_notes": _SIMULATED,
                "inference_backend": "dataset_replay_server",
                "driver": _SIMULATED,
                "operating_system": _SIMULATED,
                "filesystem": _SIMULATED,
                "other_software_stack": _SIMULATED,
                "sw_notes": _SIMULATED,
            }
        ],
        # Division / model metadata
        "division": "RDI",
        "model_id": model_id,
        "model_name": model_id,
        "model_precision": "N/A (no model executed)",
        "link_to_model": "https://github.com/mlcommons/endpoints",
        "model_notes": _SIMULATED,
        # Dataset metadata
        "dataset_id": dataset_name,
        "dataset_name": dataset_name,
        "input_token_average": isl_avg if isl_avg is not None else 0.0,
        "output_token_average": osl_avg if osl_avg is not None else 0.0,
        "dataset_type": "performance",
        "dataset_link": "tests/assets/datasets/dummy_1k.jsonl (committed)",
        "measured_accuracy_score": accuracy if accuracy is not None else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="a single run folder"
    )
    parser.add_argument(
        "--c-max",
        type=int,
        required=True,
        help="declared Maximum Supported Concurrency",
    )
    parser.add_argument(
        "--system-name", default="oracle_sim_ci", help="system_name to record"
    )
    args = parser.parse_args()

    desc = build_system_desc(args.run_dir, args.c_max, args.system_name)
    out = args.run_dir / "system_desc.json"
    out.write_text(json.dumps(desc, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {out} (isl_avg={desc['input_token_average']:.1f} "
        f"osl_avg={desc['output_token_average']:.1f} "
        f"accuracy={desc['measured_accuracy_score']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
