"""Shared pytest fixtures for submission checker tests.

Pre-built, anonymised §8.1 submission directories live under test_submissions/.
Each sub_* directory was generated from real measurement data with org names,
system names, and model HF-org prefixes removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TEST_SUBMISSIONS = Path(__file__).parent.parent / "test_submissions"


@pytest.fixture(scope="session")
def sub_a() -> Path:
    """MI355X 8-GPU, gpt-oss-120b (7 points, M=2048). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_a"


@pytest.fixture(scope="session")
def sub_b() -> Path:
    """MI355X 16-GPU, gpt-oss-120b (7 points, M=2048). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_b"


@pytest.fixture(scope="session")
def sub_c() -> Path:
    """TPU 4-chip, qwen3-coder-480b (7 points, M=512). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_c"


@pytest.fixture(scope="session")
def sub_d() -> Path:
    """TPU 8-chip, qwen3-coder-480b (8 points, M=1024). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_d"


@pytest.fixture(scope="session")
def sub_e() -> Path:
    """Gaudi DP=1, llama3-8b (11 points, M=1024). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_e"


@pytest.fixture(scope="session")
def sub_f() -> Path:
    """Gaudi DP=2, llama3-8b (11 points, M=1024). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_f"


@pytest.fixture(scope="session")
def sub_g() -> Path:
    """8-GPU vLLM, llama3-70b (10 points, M=2048). Missing LL and LT coverage."""
    return TEST_SUBMISSIONS / "sub_g"


@pytest.fixture(scope="session")
def sub_h() -> Path:
    """8-GPU SGLang, llama3-70b (10 points, M=2048). Missing LL and LT coverage."""
    return TEST_SUBMISSIONS / "sub_h"


@pytest.fixture(scope="session")
def sub_i() -> Path:
    """H200 8-GPU, deepseek-r1 (10 points, M=512). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_i"


@pytest.fixture(scope="session")
def sub_j() -> Path:
    """GB300 72-GPU, deepseek-r1 (10 points, M=16384). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_j"


@pytest.fixture(scope="session")
def valid_standardized() -> Path:
    """Fully compliant synthetic Standardized submission — should pass all checks."""
    return TEST_SUBMISSIONS / "valid_standardized"


@pytest.fixture(scope="session")
def invalid_submission() -> Path:
    """Synthetic submission with deliberate violations (3 points, failed accuracy)."""
    return TEST_SUBMISSIONS / "invalid_submission"
