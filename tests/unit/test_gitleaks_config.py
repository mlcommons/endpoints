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

"""Rule self-test for .gitleaks.toml.

Runs the real gitleaks binary with the repo config. Skipped when gitleaks is
not on PATH; the secret-scan CI workflow installs it and runs this file.

Fake credentials are generated at runtime so this file never holds a string
that the repo-wide scan or GitHub push protection would flag.
"""

import json
import random
import shutil
import string
import subprocess
import uuid
from pathlib import Path

import pytest

GITLEAKS = shutil.which("gitleaks") or ""
CONFIG = Path(__file__).resolve().parents[2] / ".gitleaks.toml"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not GITLEAKS, reason="gitleaks binary not on PATH"),
]

_rng = random.Random(20261021)
ALNUM = string.ascii_letters + string.digits
HEX = "0123456789abcdef"
URLSAFE = ALNUM + "_-"


def _fake(alphabet: str, n: int) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n))


POSITIVES = [
    (
        "db-uri-credentials",
        f"DATABASE_URL=postgresql://app:{_fake(ALNUM, 16)}@db.internal.example.com:5432/app",
    ),
    (
        "db-uri-credentials",
        f'engine = create_engine("postgresql+psycopg2://app:{_fake(ALNUM, 16)}@10.0.0.5/app")',
    ),
    (
        "db-uri-credentials",
        f"MONGO_URI=mongodb+srv://svc:{_fake(ALNUM, 20)}@cluster0.example.mongodb.net/db",
    ),
    ("neon-password", f"PGPASSWORD=npg_{_fake(ALNUM, 12)}"),
    ("supabase-token", f"SUPABASE_ACCESS_TOKEN=sbp_{_fake(HEX, 40)}"),
    ("supabase-token", f"SUPABASE_SERVICE_KEY=sb_secret_{_fake(URLSAFE, 32)}"),
    ("nvidia-api-key", f"NVIDIA_API_KEY=nvapi-{_fake(URLSAFE, 64)}"),
    ("ngc-api-key", f"NGC_API_KEY={_fake(ALNUM, 84)}"),
    (
        "ngc-api-key",
        f"ngc_key: {_fake(ALNUM, 26)}:{uuid.UUID(int=_rng.getrandbits(128))}",
    ),
    ("wandb-api-key", f"WANDB_API_KEY={_fake(HEX, 40)}"),
    ("dockerhub-pat", f"docker login -u ci -p dckr_pat_{_fake(URLSAFE, 27)}"),
]
CUSTOM_RULES = {rule for rule, _ in POSITIVES}

NEGATIVES = [
    (
        "f-string",
        'url = f"postgresql://postgres:{password}@db.example.supabase.co:5432/postgres"',
    ),
    (
        "bracket-template",
        "postgresql://postgres:[YOUR-PASSWORD]@db.example.supabase.co:5432/postgres",
    ),
    ("doc-user-pass", "postgresql://user:pass@host:5432/dbname"),
    ("doc-password-word", "postgresql://user:password@localhost/db"),
    ("env-var", "DATABASE_URL=postgresql://app:${PGPASSWORD}@localhost/app"),
    ("angle-template", "postgresql://app:<password>@localhost/app"),
    ("redacted", "postgresql://app:REDACTED@localhost/app"),
    ("no-credentials", "postgresql://localhost:5432/app"),
    ("wandb-commit-sha", f"wandb_commit: {_fake(HEX, 40)}"),
    ("ngc-unrelated-84-chars", f"checksum: {_fake(ALNUM, 84)}"),
]


def _scan(tmp_path: Path, line: str) -> set[str]:
    """Scan one file holding `line` with the repo config; return fired rule IDs."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "sample.txt").write_text(line + "\n")
    report = tmp_path / "report.json"
    subprocess.run(
        [
            GITLEAKS,
            "dir",
            str(src),
            "--config",
            str(CONFIG),
            "--report-format",
            "json",
            "--report-path",
            str(report),
            "--exit-code",
            "0",
            "--no-banner",
            "--log-level",
            "error",
        ],
        check=True,
    )
    return {finding["RuleID"] for finding in json.loads(report.read_text())}


@pytest.mark.parametrize(
    ("rule_id", "line"),
    POSITIVES,
    ids=[f"{rule}-{i}" for i, (rule, _) in enumerate(POSITIVES)],
)
def test_custom_rule_detects_its_format(tmp_path: Path, rule_id: str, line: str):
    assert rule_id in _scan(tmp_path, line)


@pytest.mark.parametrize(
    "line", [line for _, line in NEGATIVES], ids=[label for label, _ in NEGATIVES]
)
def test_custom_rules_ignore_placeholders(tmp_path: Path, line: str):
    assert not _scan(tmp_path, line) & CUSTOM_RULES


def test_default_rules_still_apply(tmp_path: Path):
    assert "github-pat" in _scan(tmp_path, f"GH_TOKEN=ghp_{_fake(ALNUM, 36)}")


def test_generic_rule_ignores_numeric_settings(tmp_path: Path):
    prose = "1 query = 1000 characters/tokens,\n`max_concurrency=100000`."
    assert "generic-api-key" not in _scan(tmp_path, prose)
