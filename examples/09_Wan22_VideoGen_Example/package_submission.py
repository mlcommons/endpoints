# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package WAN 2.2 Endpoints point runs as an upload tree for the cohort checker.

Usage:
    python package_submission.py OUT --system NAME --c-max 33 \\
        1:REPORT_DIR 4:REPORT_DIR 8:REPORT_DIR 33:REPORT_DIR:elected

Each point is CONCURRENCY:REPORT_DIR[:elected]. REPORT_DIR is the run's
report_dir. The script copies its artifacts to
OUT/results/NAME/wan-2.2-t2v-a14b/r<N>/ and writes point.yaml and
system_desc.json. Fill in division, model_precision, shared_src and
shared_docs before submitting; the runs do not record them.

The checker reads a video run's system_tps as qps x frames per clip (frames/s)
from the catalog, so result_summary.json is copied as the run wrote it.
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml
from inference_endpoint.validation.regions import Regions

MODEL = "wan-2.2-t2v-a14b"
REFERENCE = (
    "https://github.com/mlcommons/inference/tree/master/text_to_video/wan-2.2-t2v-a14b"
)


def point_doc(conc: int, region: str, elected: bool, c_max: int, cfg: dict) -> dict:
    """point.yaml facts the run's resolved config supports."""
    # e.g. mlperf-endpoints-v1.0-2026-10-C1-A
    ruleset = cfg["submission_ref"]["ruleset"]
    return {
        "concurrency": conc,
        **({"offline": "elected"} if elected else {}),
        "max_supported_concurrency": c_max,
        "load_pattern": cfg["settings"]["load_pattern"]["type"],
        "region": region,
        "model_name": cfg["submission_ref"]["model"],
        "link_to_model": REFERENCE,
        "dataset_name": next(
            d["name"] for d in cfg["datasets"] if d["type"] == "performance"
        ),
        "dataset_type": "performance",
        "dataset_link": REFERENCE,
        "seed_set": ruleset.rsplit("-", 1)[1],
        "target_cohort": ruleset.split("-v1.0-", 1)[1].rsplit("-", 1)[0],
        "warmup": {"duration_s": 0, "requests_issued": 0, "requests_completed": 0},
        "runtime_settings": cfg["settings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("points", nargs="+", help="CONCURRENCY:REPORT_DIR[:elected]")
    parser.add_argument("--system", required=True)
    parser.add_argument("--c-max", type=int, required=True)
    args = parser.parse_args()

    specs = [p.split(":") for p in args.points]
    c_min = min(int(s[0]) for s in specs)
    regions = Regions.for_bounds(c_min=c_min, c_max=args.c_max)
    for spec in specs:
        conc, report = int(spec[0]), Path(spec[1])
        region = regions.covers(conc)
        if region is None:
            raise SystemExit(f"concurrency {conc} is outside 1..{args.c_max}")
        dst = args.out / "results" / args.system / MODEL / f"r{conc}"
        dst.mkdir(parents=True, exist_ok=False)
        for src in (
            report / "config.yaml",
            report / "events.jsonl",
            report / "performance" / "result_summary.json",
        ):
            shutil.copy2(src, dst / src.name)
        accuracy = report / "accuracy" / "accuracy_results.json"
        if accuracy.exists():
            shutil.copy2(accuracy, dst / accuracy.name)
        cfg = yaml.safe_load((report / "config.yaml").read_text())
        doc = point_doc(conc, region.value, spec[2:] == ["elected"], args.c_max, cfg)
        (dst / "point.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        (dst / "system_desc.json").write_text(
            json.dumps({"max_supported_concurrency": args.c_max}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
