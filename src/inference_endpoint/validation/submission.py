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

"""Measurement points, the per-pareto derivation unit, and the uploaded tree.

Endpoints Rules source (pinned):
https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md

`Point` carries one `r<N>` run's named facts per §8.3 (`concurrency`,
`offline`) plus `system_tps` (§4.1's aggregate metric, from the run's
`result_summary.json`) and the run's own `system_desc.json` fields (§8.2.1
places one descriptor per run directory). `Pareto` is one
`results/<system>/<model>/` -- the unit every policy derivation is computed
per (§5.6 caps "a single submission's pareto". §5.4 lets different paretos
choose different `c_min_override`s). `Submission` is the uploaded tree: its
paretos and the tree nodes rows resolve facts from.

Facts are read once, here (`Point.from_node`): the tree feeds rows and `Point`
feeds the derivations, and both come from the same parsed artifacts, so no
second copy can drift from the first.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from inference_endpoint.validation.catalog import Catalog
from inference_endpoint.validation.refs import decode_number
from inference_endpoint.validation.regions import RegionKind, Regions
from inference_endpoint.validation.tree import Node, RuleScopes, SubmissionNode

RULES_URL = (
    "https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/"
    "endpoints_rules.md"
)

# §5.6: "the total number of measurement points on a single submission's
# pareto" may not exceed 32.
POINT_CAP = 32

# §5.7.2: the dedicated Offline run's system_tps must reach 0.98x the C_max
# point's.
OFFLINE_TPS_RATIO = 0.98

# §5.3: one mandatory point per concurrency region (Ultra Low, Low, Medium,
# High), three submitter's-choice points, and -- non-agentic -- one Offline
# run. §4.3's accuracy count N is the mandatory-point count (4 + the Offline
# slot); §5.3's minimum point count adds the three choice points.
MANDATORY_REGION_COUNT = 4
SUBMITTERS_CHOICE_COUNT = 3
# §5.4's four mandatory regions (Ultra Low through High); `margin` is the
# 10%-extension band, not a coverage requirement.
_MANDATORY_REGIONS: tuple[RegionKind, ...] = (
    RegionKind.ULTRA_LOW_CONCURRENCY,
    RegionKind.LOW_CONCURRENCY,
    RegionKind.MED_CONCURRENCY,
    RegionKind.HIGH_CONCURRENCY,
)

# The checker's `_TPS_TOLERANCE`: a stored throughput within 1% relative error
# of its derived value reads as consistent.
TPS_TOLERANCE = 0.01

# The checker's `_TPS_UTILIZATION_ABS_TOL`: the declared throughput
# utilization must sit within this absolute distance of its recomputed share
# of the curve's peak.
TPS_UTILIZATION_ABS_TOL = 0.1


def _derived_system_tps(value: Mapping[str, Any]) -> float | None:
    """The whole-run throughput a summary's own numbers imply, or `None`.

    `output_sequence_lengths.total` over `duration_ns` seconds -- the derived
    side of Section 9.1's throughput checks. A summary missing either number,
    or carrying a non-finite or non-positive one, yields `None` and its row is
    left to the presence refusals.

    Args:
        value (Mapping[str, Any]): One `result_summary.json` document.
    Returns:
        float | None: The derived tokens per second, `None` when the summary's
            own numbers cannot carry one.
    """
    totals = value.get("output_sequence_lengths")
    total = decode_number(
        totals.get("total") if isinstance(totals, Mapping) else None, "float"
    )
    duration_ns = decode_number(value.get("duration_ns"), "float")
    numbers = (total, duration_ns)
    if (
        not all(
            isinstance(number, int | float) and not isinstance(number, bool)
            for number in numbers
        )
        or not all(math.isfinite(float(number)) for number in numbers)
        or duration_ns <= 0
    ):
        return None
    seconds = float(duration_ns) / 1e9
    if seconds <= 0:
        # A subnormal nanosecond count rounds to zero seconds: no throughput
        # can be derived, and dividing by it would raise.
        return None
    return float(total) / seconds


# The bookkeeping keys an accuracy entry carries beside its metrics: never a
# metric, whatever their value type.
_ACCURACY_META_KEYS: frozenset[str] = frozenset(
    {
        "dataset_name",
        "num_samples",
        "n_repeats",
        "unit_samples",
        "total_samples",
        "osl_tokenize_s",
    }
)


def _accuracy_entries(root: Any) -> list[Mapping[str, Any]]:
    """The per-dataset entries of an `accuracy_scores` value.

    The round's checker reads a mapping keyed by dataset name; the client's own
    writer emits a list of entries carrying `dataset_name`. Both read the same.

    Args:
        root (Any): The `accuracy_results.json` document's `accuracy_scores`.
    Returns:
        list[Mapping[str, Any]]: Each entry, mapping-shaped ones only.
    """
    if isinstance(root, Mapping):
        return [entry for entry in root.values() if isinstance(entry, Mapping)]
    if isinstance(root, Sequence) and not isinstance(root, str | bytes):
        return [entry for entry in root if isinstance(entry, Mapping)]
    return []


def _accuracy_document_entries(
    document: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """One `accuracy_results.json` document's per-dataset entries.

    Two shapes ship: the client's `{"accuracy_scores": [ ... ]}` (or a mapping
    under that key), and the canonical document keyed by dataset name at its top
    level. One reader, so `_accuracy_carrying_points`, the gate and the
    presence row all count the same entries.
    """
    if "accuracy_scores" in document:
        return _accuracy_entries(document.get("accuracy_scores"))
    return [
        entry
        for key, entry in document.items()
        if key not in _ACCURACY_META_KEYS and isinstance(entry, Mapping)
    ]


def _accuracy_entry_scores(entry: Mapping[str, Any]) -> dict[str, float]:
    """One entry's `{metric: value}`.

    The checker reads three shapes: a `score` mapping of named metrics, a
    scalar `score` (kept under its own name), or the metrics as direct entry
    keys, with the bookkeeping keys excluded. A value that decodes to no number
    is skipped, as the checker logs and skips it; a non-finite one is kept, for
    the gate to judge rather than gate to a silent pass.

    Args:
        entry (Mapping[str, Any]): One dataset entry.
    Returns:
        dict[str, float]: The entry's numeric metrics, by name.
    """
    raw = entry.get("score")
    if isinstance(raw, Mapping):
        items = list(raw.items())
    elif raw is not None:
        items = [("score", raw)]
    else:
        items = [
            (key, value)
            for key, value in entry.items()
            if key not in _ACCURACY_META_KEYS
        ]
    scores: dict[str, float] = {}
    for key, value in items:
        decoded = decode_number(value, "float")
        if isinstance(decoded, int | float) and not isinstance(decoded, bool):
            scores[str(key)] = float(decoded)
    return scores


def _accuracy_weight(entry: Mapping[str, Any]) -> float | None:
    """One entry's aggregation weight: its sample count, `None` when unstated.

    A non-finite count is no weight: an infinite one would poison the mean, so
    the plain-mean fallback takes over.
    """
    for key in ("num_samples", "unit_samples"):
        weight = decode_number(entry.get(key), "float")
        if (
            isinstance(weight, int | float)
            and not isinstance(weight, bool)
            and math.isfinite(float(weight))
        ):
            return float(weight)
    return None


def _accuracy_issued(entry: Mapping[str, Any]) -> float | None:
    """One entry's issued sample count, `None` when it states none.

    `num_samples x max(n_repeats, 1)`, the checker's own reading of the repeats
    a dataset was run with; the client's `total_samples` is that product
    already.
    """
    total = decode_number(entry.get("total_samples"), "float")
    if isinstance(total, int | float) and not isinstance(total, bool):
        return float(total)
    samples = _accuracy_weight(entry)
    if samples is None:
        return None
    repeats = decode_number(entry.get("n_repeats"), "float")
    if not isinstance(repeats, int | float) or isinstance(repeats, bool):
        repeats = 1.0
    return samples * max(float(repeats), 1.0)


def _accuracy_bounds(targets: Any) -> dict[str, tuple[float, float | None]]:
    """The catalog's per-metric gate bounds, scaled by each entry's `factor`.

    A `lower` entry gates below (`lower x factor`); a `band` gates on both
    sides, its upper bound widened by the same tolerance (`upper x (2 - factor)`
    -- the checker's own `x 0.9` / `x 1.1` pair for a `factor: 0.9` band).

    Args:
        targets (Any): The catalog model's `accuracy_targets` mapping.
    Returns:
        dict[str, tuple[float, float | None]]: `metric -> (lower, upper)`,
            `upper` `None` for a one-sided gate; empty when nothing declares a
            bound.
    """
    if not isinstance(targets, Mapping):
        return {}
    bounds: dict[str, tuple[float, float | None]] = {}
    for metric, declared in targets.items():
        if not isinstance(declared, Mapping):
            continue
        raw_factor = decode_number(declared.get("factor"), "float")
        factor = (
            float(raw_factor)
            if isinstance(raw_factor, int | float) and not isinstance(raw_factor, bool)
            else 1.0
        )
        band = declared.get("band")
        if (
            isinstance(band, Sequence)
            and not isinstance(band, str | bytes)
            and len(band) == 2
        ):
            lower = decode_number(band[0], "float")
            upper = decode_number(band[1], "float")
            if all(
                isinstance(bound, int | float) and not isinstance(bound, bool)
                for bound in (lower, upper)
            ):
                bounds[str(metric)] = (
                    float(lower) * factor,
                    float(upper) * (2.0 - factor),
                )
            continue
        lower = decode_number(declared.get("lower"), "float")
        if isinstance(lower, int | float) and not isinstance(lower, bool):
            bounds[str(metric)] = (float(lower) * factor, None)
    return bounds


def _weighted_mean(pairs: Sequence[tuple[float, float | None]]) -> float:
    """The sample-weighted mean of `(value, weight)` pairs.

    The plain mean when any weight is missing or the weights sum to zero, the
    checker's own fallback.
    """
    weights = [weight for _, weight in pairs]
    if all(weight is not None for weight in weights):
        total = sum(weight for weight in weights if weight is not None)
        if total > 0:
            return (
                sum(value * weight for value, weight in pairs if weight is not None)
                / total
            )
    return sum(value for value, _ in pairs) / len(pairs)


# §8.5 describes one system per curve, but §8.1 calls each point's copy of the
# system description the "framework/parallelism/precision for this point", and
# §4.5.3's worked example changes the replica size between points: these are
# the fields a curve's points may legitimately spell differently.
_PER_POINT_SYSTEM_DESC_FIELDS: frozenset[str] = frozenset(
    {
        "tps_utilization",
        "tensor_parallel",
        "pipeline_parallel",
        "expert_parallel",
        "data_parallel",
        "disaggregated",
        "batch",
        "config_summary",
        "config_summary_notes",
    }
)


def _resolves_shared_path(root: Path, value: str) -> Path | None:
    """`value` as Section 9.1's shared pointer under `root`, or `None`.

    The checker's rule: a non-blank POSIX-style relative pointer, free of `..`,
    no absolute or `~` prefix, naming an existing directory below the root.
    """
    raw = value.strip()
    if not raw or raw.startswith(("/", "~")):
        return None
    parts = [part for part in raw.replace("\\", "/").split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return None
    candidate = root.joinpath(*parts)
    return candidate if candidate.is_dir() else None


# Section 2.9.4's lead time: two cohorts after the entry's approval.
_DRAFTER_APPROVAL_LEAD_COHORTS = 2


def _advance_cohort(cohort: str, steps: int) -> str | None:
    """`cohort` advanced by `steps` publications, or `None` when it is no cohort.

    The calendar's own order: `C0` -> `C1` -> the next month's `C0`. Zero steps
    is the spelling check.
    """
    match = re.fullmatch(r"(\d{4})-(\d{2})-C([01])", cohort)
    if match is None:
        return None
    year, month, index = (int(part) for part in match.groups())
    for _ in range(steps):
        if index == 0:
            index = 1
        else:
            index = 0
            month += 1
            if month > 12:
                month = 1
                year += 1
    return f"{year:04d}-{month:02d}-C{index}"


def _drafter_entry_matches(
    entry: Mapping[str, Any], declared: Mapping[str, Any]
) -> bool:
    """Whether one approved entry names the declared drafter (§9.1's two forms).

    A weight-identified entry matches by `weight_checksum`; a
    configuration-identified one by `target_checksum` plus its non-empty
    `configuration`. A model id alone never matches, as Section 2.9.4's
    checksum identification intends.
    """
    weight = entry.get("weight_checksum")
    if isinstance(weight, str) and weight:
        return declared.get("weight_checksum") == weight
    target = entry.get("target_checksum")
    configuration = entry.get("configuration")
    if isinstance(target, str) and target:
        return (
            declared.get("target_checksum") == target
            and isinstance(configuration, Mapping)
            and bool(configuration)
            and declared.get("configuration") == configuration
        )
    return False


def normalize_offline(declared: Any) -> Any:
    """The `offline` declaration as `Point.offline` carries it.

    Endpoints Policy Section 8.3's field is a three-state declaration:
    absent-or-`none` is "no declaration", so both spellings normalize together.

    Args:
        declared (Any): The raw `offline` value read from a run's `point.yaml`.
    Returns:
        Any: `"dedicated"` or `"elected"` when declared. `None` for
            absent-or-`none`.
    """
    return None if declared in (None, "none") else declared


def _mapping_artifact(node: Node, name: str) -> Mapping[str, Any]:
    """The parsed mapping artifact `name` of `node`, or a refusal naming it."""
    value, error = node.read(name)
    if error is not None or not isinstance(value, Mapping):
        raise ValueError(
            f"{node.dir / name} cannot carry its facts: {error or 'not a mapping'}"
        )
    return value


def _int_fact(source: Mapping[str, Any], name: str, where: str) -> int:
    """`source[name]` as the integer fact `where` documents.

    A numeric string reads as the number it spells (`refs.decode_number`), the
    same rule the ref table applies to the leaves it declares. The value must
    survive `float()` too: the region algorithm computes in floats, so an
    integer Python cannot widen would crash a derivation instead of refusing the
    upload.
    """
    value = decode_number(source.get(name), "int")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} needs an integer {name}")
    try:
        float(value)
    except OverflowError:
        raise ValueError(f"{where} {name} {value} is outside float range") from None
    return value


def _finite_fact(source: Mapping[str, Any], name: str, where: str) -> float:
    """`source[name]` as the finite float fact `where` documents.

    `NaN`/`Infinity` decode from `result_summary.json` and a huge integer cannot
    widen to `float`: either would reach a derivation as a number that compares
    quietly (or raises uncaught), so both are refused at load, where the refusal
    is a report line instead of a crash mid-derivation. A numeric string reads
    as the number it spells (`refs.decode_number`), the rule the ref table
    applies to the leaves it declares.
    """
    value = decode_number(source.get(name), "float")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where} needs a numeric {name}")
    try:
        converted = float(value)
    except OverflowError:
        raise ValueError(f"{where} {name} {value} is outside float range") from None
    if not math.isfinite(converted):
        raise ValueError(f"{where} {name} needs a finite value, got {value!r}")
    return converted


def _descriptor_facts(system_desc: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    """`system_desc` with its numeric facts decoded, as `Point` carries it.

    The artifact document itself stays as written (the ref table reads it), but
    every consumer of the carried descriptor sees numbers: the required maximum
    is an `int` (the pareto shape check and the region bounds do arithmetic on
    it), and the optional `c_min_override` decodes where it is spelled as a
    numeric string. Any other value stays as it is — a present `null` override
    keeps §5.4's own refusal naming it, rather than this pass restating it.

    Example:

        _descriptor_facts({"max_supported_concurrency": "8192"}, "system_desc.json")
        # {"max_supported_concurrency": 8192}

    Args:
        system_desc (Mapping[str, Any]): The run's parsed `system_desc.json`.
        where (str): The document's path, for the refusal messages.
    Returns:
        Mapping[str, Any]: A copy carrying the decoded facts.
    """
    carried = dict(system_desc)
    carried["max_supported_concurrency"] = _int_fact(
        system_desc, "max_supported_concurrency", where
    )
    override = system_desc.get("c_min_override")
    if isinstance(override, str):
        carried["c_min_override"] = decode_number(override, "int")
    return carried


@dataclass(frozen=True)
class Point:
    """One measurement point's named facts (Endpoints Policy §8.3 + §4.1).

    Args:
        offline (Literal["dedicated", "elected"] | None): The field's three-state
            declaration verbatim (`"dedicated"`, `"elected"`, or `None` for
            absent-or-`none`): a point with `offline is None` is neither an elected nor
            a dedicated Offline run.
        concurrency (int): The run's declared point concurrency (§8.3).
        system_tps (float): §4.1's aggregate metric in the run's
            `result_summary.json` (§8.1).
        system_desc (Mapping[str, Any]): This run's own `system_desc.json` fields
            (§8.2.1).
        node (Node | None): The run directory's tree node. `None` on a
            derivation-only point (built by hand for derivations over named values, not
            walked from the tree).
        model_name (str | None): The declared `model_name` (§8.3): the key the
            catalog's agentic flag is looked up by. `None` when the artifact omits it
            or spells another type. (Default: None)
        load_pattern (str | None): The declared `runtime_settings.load_pattern`,
            defaulted to `concurrency` where the block omits it (the schema's own
            default). (Default: None)
        dataset (str | None): The declared `dataset` id. (Default: None)
        seed_set (str | None): The declared `seed_set` id. (Default: None)
        region (str | None): The declared `region` (§8.3) verbatim: a band name
            or `submitters_choice`. `None` when the artifact omits it or spells
            another type. (Default: None)
    """

    offline: Literal["dedicated", "elected"] | None
    concurrency: int
    system_tps: float
    system_desc: Mapping[str, Any] = field(hash=False)
    # A derivation-only Point carries no tree link; `Finding.collect` refuses
    # one without its point node.
    node: Node | None = None
    model_name: str | None = None
    load_pattern: str | None = None
    dataset: str | None = None
    seed_set: str | None = None
    region: str | None = None

    def __post_init__(self) -> None:
        if self.offline not in (None, "dedicated", "elected"):
            raise ValueError(
                f"offline must be 'dedicated', 'elected' or null, got {self.offline!r}"
            )

    @classmethod
    def from_node(cls, node: Node) -> Point:
        """The `Point` one `r<N>` run directory's artifacts declare.

        Per §8.2.1/§8.3 the run carries its own `system_desc.json` and
        `point.yaml`. `system_tps` is §4.1's aggregate metric in the run's
        `result_summary.json` (§8.1), not a `point.yaml` field.

        Args:
            node (Node): The `r<N>` run directory's tree node.
        Returns:
            Point: The run's named facts, with `node` carried on the point.
        Raises:
            ValueError: When a named source is missing, mistyped, or a number no
                arithmetic can carry (non-finite, or outside `float` range). Each named
                source is enforced here, at load: a missing or mistyped fact names its
                artifact rather than reaching a row as an absent value.
        """
        config = _mapping_artifact(node, "point.yaml")
        concurrency = _int_fact(config, "concurrency", f"{node.dir}/point.yaml")
        declared_name = config.get("model_name")
        offline = normalize_offline(config.get("offline"))
        if offline not in (None, "dedicated", "elected"):
            raise ValueError(
                f"{node.dir}/point.yaml offline must be 'dedicated', 'elected' "
                f"or null, got {offline!r}"
            )
        system_desc = _descriptor_facts(
            _mapping_artifact(node, "system_desc.json"),
            f"{node.dir}/system_desc.json",
        )
        summary = _mapping_artifact(node, "result_summary.json")
        system_tps = _finite_fact(
            summary, "system_tps", f"{node.dir}/result_summary.json"
        )
        runtime_settings = config.get("runtime_settings")
        pattern = (
            runtime_settings.get("load_pattern", "concurrency")
            if isinstance(runtime_settings, Mapping)
            else None
        )
        return cls(
            offline=offline,
            concurrency=concurrency,
            system_tps=system_tps,
            system_desc=system_desc,
            node=node,
            model_name=declared_name if isinstance(declared_name, str) else None,
            load_pattern=pattern if isinstance(pattern, str) else None,
            dataset=(
                config.get("dataset")
                if isinstance(config.get("dataset"), str)
                else None
            ),
            seed_set=(
                config.get("seed_set")
                if isinstance(config.get("seed_set"), str)
                else None
            ),
            region=(
                config.get("region") if isinstance(config.get("region"), str) else None
            ),
        )

    def point_power_kw(
        self,
        provisioned_power_kw: float,
        engaged_nodes: int,
        provisioned_nodes: int,
    ) -> float:
        """The Endpoints Policy Section 4.5.3 point denominator `provisioned_power_kw x Y/N`.

        Section 4.5.2/4.5.3's normalization denominator per measurement point:
        provisioned power scaled by the node fraction `Y_s / N_s` engaged at
        this point (whole nodes rounded up from accelerators). Rounded to two
        decimal places, once at the final figure (§4.5's Units-and-rounding
        rule). `engaged_nodes` (Y) and `provisioned_nodes` (N) are supplied by
        the caller until the fuller `nodes_used` carry lands on `Point` itself.

        Example of Appendix C.3's golden row:

            point = Point(offline=None, concurrency=32, system_tps=6000.0, system_desc={})
            point.point_power_kw(152.10, 16, 18)  # 135.2 == round(152.10 * 16 / 18, 2)

        With `engaged_nodes == provisioned_nodes` this reduces to `round(provisioned_power_kw, 2)`
        (the `152.10 x 18/18 == 152.1` row).

        Args:
            provisioned_power_kw (float): The system's provisioned power, in kW.
            engaged_nodes (int): `Y`: the nodes engaged at this point (whole
                nodes rounded up from accelerators).
            provisioned_nodes (int): `N`: the system's provisioned node count.
        Returns:
            float: The `provisioned_power_kw x Y/N` denominator, rounded to two
                decimal places.
        """
        return round(provisioned_power_kw * engaged_nodes / provisioned_nodes, 2)


@dataclass(frozen=True)
class Pareto:
    """One `results/<system>/<model>/`: the unit every policy derivation is per.

    Args:
        system_desc (Mapping[str, Any]): The declared facts shared by the pareto's
            points: `max_supported_concurrency` and `c_min_override`, both §8.3 fields.
        points (Sequence[Point]): The ordered `Point`s of one pareto.
        node (Node | None): The tree's pareto node when built from an upload
            (`Pareto.from_nodes`). `None` for a hand-built derivation-only Pareto
            (`Pareto(system_desc, points)`), whose points carry no node and which
            `Finding.collect` refuses to walk.
    """

    system_desc: Mapping[str, Any] = field(hash=False)
    points: Sequence[Point]
    # The tree's pareto node when built from an upload (`Pareto.from_nodes`);
    # `None` for a derivation-only hand-built Pareto.
    node: Node | None = None

    def __post_init__(self) -> None:
        # frozen dataclass: object.__setattr__ is the only sanctioned mutation.
        object.__setattr__(self, "points", tuple(self.points))
        self._validate_shape()

    @classmethod
    def from_nodes(cls, node: Node) -> Pareto:
        """The Pareto one `<system>/<model>` node's run directories declare.

        The pareto's §5.5 regions latch onto its tree node here, so the
        `regions` namespace reaches every row below it (`Node.scopes` reads the
        nearest ancestor's regions).

        Args:
            node (Node): The `<system>/<model>` pareto node.
        Returns:
            Pareto: The pareto of the node's run directories, with its §5.5
                regions latched onto `node`.
        """
        points = tuple(Point.from_node(child) for child in node.children)
        pareto = cls(_shared_facts(points, node.dir), points, node=node)
        node.regions = pareto.regions()
        return pareto

    def _validate_shape(self) -> None:
        """Fail loud on shapes the derivations would trip over mid-report.

        These are rule violations, not parse errors (Endpoints Policy §9.1's
        Offline-point-present and Max-concurrency-declared rows). Each message
        names its row until they reach the report as findings.

        Raises:
            ValueError: When the pareto has no measurement point. When its points
                do not declare one integer `max_supported_concurrency` (§9.1, Max
                concurrency declared). When more than one point carries an
                `offline` declaration (§9.1, Offline point present). When an
                `elected` declaration sits off the C_max point (§9.1, Offline point
                present). When a dedicated Offline run lacks exactly one C_max
                point (§9.1, Offline ordering). When two fixed-concurrency points
                declare the same concurrency (§5.3, Minimum Submission
                Requirements; the dedicated Offline run is exempt).
        """
        if not self.points:
            raise ValueError("a pareto needs at least one measurement point")
        c_max = self.system_desc.get("max_supported_concurrency")
        if (
            isinstance(c_max, bool)
            or not isinstance(c_max, int)
            or self.points[0].system_desc.get("max_supported_concurrency") != c_max
        ):
            raise ValueError(
                "a pareto's points must declare one integer "
                "max_supported_concurrency (Endpoints Policy §8.3; §9.1, "
                "Max concurrency declared)"
            )
        declared, elected, dedicated = self._offline_partition()
        if len(declared) > 1:
            raise ValueError(
                "exactly one point may carry an `offline` declaration "
                "(Endpoints Policy §9.1, Offline point present)"
            )
        if elected:
            point = elected[0]
            if point.concurrency != c_max:
                raise ValueError(
                    "an `elected` declaration must sit on the C_max point "
                    "(Endpoints Policy §9.1, Offline point present)"
                )
        if dedicated:
            rivals = [
                p
                for p in self.points
                if p is not dedicated[0] and p.concurrency == c_max
            ]
            if len(rivals) != 1:
                raise ValueError(
                    "a dedicated Offline run needs exactly one C_max point "
                    "(Endpoints Policy §9.1, Offline ordering)"
                )

        seen: dict[int, Point] = {}
        for point in self.points:
            if point.offline == "dedicated":
                continue
            if point.concurrency in seen:
                raise ValueError(
                    f"a pareto's fixed-concurrency points must be distinct: "
                    f"concurrency {point.concurrency} is declared more than once "
                    "(Endpoints Policy §5.3, Minimum Submission Requirements; "
                    "the dedicated Offline run is exempt)"
                )
            seen[point.concurrency] = point

    def _offline_partition(self) -> tuple[list[Point], list[Point], list[Point]]:
        """`(declared, elected, dedicated)` over the `offline` declaration field."""
        declared = [p for p in self.points if p.offline is not None]
        elected = [p for p in declared if p.offline == "elected"]
        dedicated = [p for p in declared if p.offline == "dedicated"]
        return declared, elected, dedicated

    def _c_max_point(self, exclude: Point | None = None) -> Point:
        """The unique non-`exclude` point at `concurrency == M` (validated at build)."""
        c_max = self.system_desc["max_supported_concurrency"]
        return next(
            p for p in self.points if p is not exclude and p.concurrency == c_max
        )

    def regions(self) -> Regions:
        """This pareto's §5.5 regions (`c_min_override`-aware, one source per pareto)."""
        return Regions.for_pareto(
            self.system_desc, [point.concurrency for point in self.points]
        )

    def point_cap(self) -> bool:
        """Whether Endpoints Policy §5.6's point cap holds (<= 32 points per pareto).

        `Pareto({...}, [...32 points]).point_cap()` -> True (verified:
        `32 <= 32`). 33 points -> False.

        Returns:
            bool: Whether the pareto holds at most `POINT_CAP` (32) measurement
                points. The dedicated Offline point counts.
        """
        return len(self.points) <= POINT_CAP

    def out_of_range_points(self, regions: Regions) -> tuple[Point, ...]:
        """Fixed-concurrency points outside `regions.coverage` (§9.1's Concurrency-in-range row).

        "Flag out-of-range points" (§9.1's Concurrency-in-range row of the
        compliance checklist) flags a point whose concurrency falls outside
        every valid region, the 10% High Concurrency margin included. A
        dedicated Offline point is exempt: it "is not a fixed-concurrency
        point" (§5.7.2) and its `concurrency` is the performance dataset's
        cardinality (§5.7), so no region judges it.

        Args:
            regions (Regions): The pareto's §5.5 `Regions`, taken rather than
                recomputed so one derivation source stays.
        Returns:
            tuple[Point, ...]: The fixed-concurrency points whose concurrency
                falls outside `regions.coverage`. Empty when every fixed-concurrency
                point is in range.
        """
        return tuple(
            p
            for p in self.points
            if p.offline != "dedicated" and p.concurrency not in regions.coverage()
        )

    def offline_ordering(self) -> bool | None:
        """Whether the dedicated Offline run meets Endpoints Policy §5.7.2's two clauses.

        `system_tps(Offline) >= OFFLINE_TPS_RATIO * system_tps` at the C_max
        point, and `concurrency(Offline) >= C_max` (the declared
        `max_supported_concurrency`). Mixed/multiple declarations and a missing
        C_max point are refused at build (`Pareto._validate_shape`), so this
        derivation's value is always meaningful.

        Returns:
            bool | None: True when both §5.7.2 clauses hold. False when either
                fails. `None` when not applicable: the point is `elected`, or no
                `offline` declaration is present at all (agentic benchmarks).
        """
        declared, elected, _ = self._offline_partition()
        if elected or not declared:
            return None
        offline_point = declared[0]
        c_max = self.system_desc["max_supported_concurrency"]
        c_max_point = self._c_max_point(exclude=offline_point)
        return (
            offline_point.system_tps >= OFFLINE_TPS_RATIO * c_max_point.system_tps
            and offline_point.concurrency >= c_max
        )

    def agentic(self) -> bool:
        """Whether this pareto is an agentic benchmark, from the catalog first.

        §5.7/§4.3 care whether the benchmark is agentic (the Offline point and
        the fifth accuracy run apply to non-agentic only). The declared
        `model_name` keys the catalog's `agentic` flag, so a missing Offline
        declaration is a finding rather than a redefinition of the minimum. A
        pareto with no catalog entry (no ruleset, an unknown model, or a
        hand-built pareto) falls back to the declared shape: an `offline`
        declaration reads as non-agentic, its absence as agentic.

        Returns:
            bool: The catalog's flag for the declared model, else the declared
                shape's reading.
        """
        catalog = self._catalog()
        model_name = self._model_name()
        if catalog is not None and model_name is not None:
            entry = catalog.models.get(model_name)
            if isinstance(entry, Mapping):
                return bool(entry.get("agentic", False))
        return not any(p.offline is not None for p in self.points)

    def _catalog(self) -> Catalog | None:
        """The ruleset catalog this pareto's tree was built with, if any."""
        if self.node is None:
            return None
        return self.node.root().ruleset

    def _model_name(self) -> str | None:
        """The first `model_name` the pareto's points declare."""
        return next((p.model_name for p in self.points if p.model_name), None)

    def model_name_admitted(self) -> bool:
        """Whether every declared `model_name` is one the round's ruleset admits.

        §3.2's benchmark-model list is the catalog's: `ruleset.models` names the
        admitted models, and a name outside it is a finding -- a curve whose
        points declare different names fails on whichever is not admitted.
        Without a catalog the answer is `False` and the row's `when` gates it
        off (there is no list to judge against); with a catalog and no point
        declaring a name it is also `False`, and the disclosure rows report the
        absence.

        Returns:
            bool: Whether every declared model name is in the catalog.
        """
        catalog = self._catalog()
        if catalog is None:
            return False
        declared = [point.model_name for point in self.points if point.model_name]
        return bool(declared) and all(name in catalog.models for name in declared)

    def point_count(self) -> int:
        """The pareto's measurement-point count (§5.3/§9.1's Point count)."""
        return len(self.points)

    def uncovered_regions(self) -> tuple[str, ...]:
        """The §5.4 mandatory regions no point lands in (coverage findings).

        "One point in each of the concurrency regions" (§5.3): a region is
        covered when a fixed-concurrency point's concurrency falls in it. The
        dedicated Offline run covers nothing (§5.7.2: it does not satisfy any
        region-coverage requirement) and the `margin` band is not a coverage
        region, so neither can satisfy the requirement.

        Returns:
            tuple[str, ...]: One `region` wire value per uncovered mandatory
                region, empty when every one is covered.
        """
        regions = self.regions()
        covered: set[RegionKind] = set()
        for point in self.points:
            if point.offline == "dedicated":
                continue
            kind = regions.covers(point.concurrency)
            if kind is not None:
                covered.add(kind)
        return tuple(kind.value for kind in _MANDATORY_REGIONS if kind not in covered)

    def offline_points(self) -> tuple[Point, ...]:
        """The points carrying an `offline` declaration (§5.7's presence rule).

        Refused at build when more than one carries one, so a non-agentic
        pareto reads `1` here and an agentic pareto's ideal is `0`.
        """
        return tuple(self._offline_partition()[0])

    def conflicting_load_patterns(self) -> tuple[str, ...]:
        """The distinct load patterns when the type-voting points disagree (§8.5).

        §6.1 gives a dedicated Offline run its own pattern (`max_throughput`), so
        it is left out of the vote; an elected point is an ordinary
        fixed-concurrency run and votes. Empty when the vote is unanimous or no
        voting point declares a pattern.

        Returns:
            tuple[str, ...]: The distinct patterns, sorted, when more than one
                is in play; otherwise empty.
        """
        patterns = {
            point.load_pattern
            for point in self.points
            if point.offline != "dedicated" and point.load_pattern is not None
        }
        return tuple(sorted(patterns)) if len(patterns) > 1 else ()

    def conflicting_model_names(self) -> tuple[str, ...]:
        """The distinct declared `model_name`s when the points disagree (§9.1).

        Returns:
            tuple[str, ...]: The distinct names, sorted, when more than one is
                declared; otherwise empty.
        """
        names = {point.model_name for point in self.points if point.model_name}
        return tuple(sorted(names)) if len(names) > 1 else ()

    def conflicting_datasets(self) -> tuple[str, ...]:
        """The distinct `dataset` ids when the points disagree (§9.1).

        A point that declares no `dataset` reads as the schema's own default
        (the empty string), so one declared id beside an omitted key is a
        disagreement, exactly as the round's checker reads it.

        Returns:
            tuple[str, ...]: The distinct ids, sorted, when more than one is in
                play; otherwise empty.
        """
        datasets = {point.dataset or "" for point in self.points}
        return tuple(sorted(datasets)) if len(datasets) > 1 else ()

    def conflicting_seed_sets(self) -> tuple[str, ...]:
        """The distinct `seed_set` ids when the points disagree (§9.1).

        A point with no declaration is left out: absence is the disclosure
        rows' report, not a disagreement.

        Returns:
            tuple[str, ...]: The distinct ids, sorted, when more than one is
                declared; otherwise empty.
        """
        ids = {point.seed_set for point in self.points if point.seed_set is not None}
        return tuple(sorted(ids)) if len(ids) > 1 else ()

    def inconsistent_system_desc_keys(self) -> tuple[str, ...]:
        """Keys whose value differs between the curve's point descriptions (§8.5).

        Every point of a curve must describe the same system. Each point's
        `system_desc.json` is compared to the first's, with the fields §8.1 and
        §4.5.3 leave per-point (`_PER_POINT_SYSTEM_DESC_FIELDS`) excluded.

        Returns:
            tuple[str, ...]: The differing keys, sorted; empty when every
                description reads the same (or none is parsable).
        """
        identities: list[Mapping[str, Any]] = []
        for point in self.points:
            if point.node is None:
                continue
            value, error = point.node.read("system_desc.json")
            if error is not None or not isinstance(value, Mapping):
                continue
            identities.append(
                {
                    key: item
                    for key, item in value.items()
                    if key not in _PER_POINT_SYSTEM_DESC_FIELDS
                }
            )
        if not identities:
            return ()
        first = identities[0]
        differing: set[str] = set()
        for identity in identities[1:]:
            for key in set(identity) | set(first):
                if not _same_value(identity.get(key), first.get(key)):
                    differing.add(key)
        return tuple(sorted(differing))

    def accounting_mismatch_points(self) -> tuple[int, ...]:
        """The points whose sample counts do not add up (§14, Metric Consistency).

        `n_samples_completed + n_samples_failed` must equal `n_samples_issued`.
        A missing `n_samples_failed` reads as `0` (the summary schema's own
        default), and the counts decode through `decode_number`, so a
        stringified count reads as its number; a point whose summary omits the
        other two counts is left to the required-fact refusals.

        Returns:
            tuple[int, ...]: The concurrency of each inconsistent point, empty
                when every summary adds up.
        """
        mismatched: list[int] = []
        for point in self.points:
            if point.node is None:
                continue
            value, error = point.node.read("result_summary.json")
            if error is not None or not isinstance(value, Mapping):
                continue
            issued = decode_number(value.get("n_samples_issued"), "int")
            completed = decode_number(value.get("n_samples_completed"), "int")
            failed = decode_number(value.get("n_samples_failed", 0), "int")
            counts = (issued, completed, failed)
            if not all(
                isinstance(count, int) and not isinstance(count, bool)
                for count in counts
            ):
                continue
            if completed + failed != issued:
                mismatched.append(point.concurrency)
        return tuple(mismatched)

    def system_tps_mismatch_points(self) -> tuple[int, ...]:
        """The points whose stored throughput misses `tokens / duration` (§9.1).

        Section 14/9.1: `system_tps` must equal the whole run's
        `output_sequence_lengths.total` over its `duration_ns` seconds. The
        stored side is the whole-run reading -- `full_run_system_tps` where the
        tree aliased the client's `tps`, the document's own `system_tps`
        otherwise -- never Section 4.4's steady-window alias. Within
        `TPS_TOLERANCE` relative error the two agree, as the checker reads it.

        Returns:
            tuple[int, ...]: The concurrency of each mismatching point, empty
                when every stored throughput agrees.
        """
        mismatched: list[int] = []
        for point in self.points:
            if point.node is None:
                continue
            value, error = point.node.read("result_summary.json")
            if error is not None or not isinstance(value, Mapping):
                continue
            stored = decode_number(
                value.get("full_run_system_tps", value.get("system_tps")), "float"
            )
            derived = _derived_system_tps(value)
            if not isinstance(stored, int | float) or isinstance(stored, bool):
                continue
            if not math.isfinite(float(stored)):
                continue
            if derived is None:
                # A declared throughput the summary's own numbers cannot derive
                # (a missing, null or non-positive duration) is unverifiable,
                # and unverifiable must not read as consistent.
                mismatched.append(point.concurrency)
                continue
            if abs(float(stored) - derived) / max(abs(derived), 1e-9) > TPS_TOLERANCE:
                mismatched.append(point.concurrency)
        return tuple(mismatched)

    def misplaced_region_points(self) -> tuple[int, ...]:
        """Points whose declared region is not the one their concurrency computes.

        Section 8.3's `region` is optional and vocabulary-only per point, while
        Sections 5.5/5.7 derive the boundaries from the curve's own
        `C_min`/`C_max`: the checker's `region-placement` compares each declared
        region against the band its concurrency classifies into and warns on a
        mismatch. A point that declares no region, declares
        `submitters_choice`, or sits outside every band (the in-range row's own
        finding) is not counted here.

        Returns:
            tuple[int, ...]: The concurrency of each misplaced point, empty when
                every declared region agrees with its band.
        """
        regions = self.regions()
        misplaced: list[int] = []
        for point in self.points:
            declared = point.region
            if declared is None or declared == "submitters_choice":
                continue
            computed = regions.covers(point.concurrency)
            if computed is None:
                continue
            if computed.value != declared:
                misplaced.append(point.concurrency)
        return tuple(misplaced)

    def tps_utilization_mismatch_points(self) -> tuple[int, ...]:
        """Points whose declared throughput utilization misses its curve share.

        Section 8.2's `system_desc.tps_utilization` normalises a point against
        the peak `system_tps` of its own curve, and Section 9.1 makes the value
        derivable: the checker recomputes `system_tps / max(system_tps)` from
        the summaries -- the derived whole-run throughputs, not the stored
        field -- and accepts a declared value within `TPS_UTILIZATION_ABS_TOL`.
        A point whose summary cannot carry a throughput is left to the presence
        refusals; the peak is taken over every derivable throughput, declared
        utilization or not.

        Returns:
            tuple[int, ...]: The concurrency of each mismatching point, empty
                when every declared utilization agrees.
        """
        throughputs: list[tuple[Point, float]] = []
        for point in self.points:
            if point.node is None:
                continue
            value, error = point.node.read("result_summary.json")
            derived = (
                _derived_system_tps(value)
                if error is None and isinstance(value, Mapping)
                else None
            )
            if derived is not None:
                throughputs.append((point, derived))
        peak = max((derived for _, derived in throughputs), default=0.0)
        if peak <= 0:
            return ()
        misses: list[int] = []
        for point, derived in throughputs:
            util = decode_number(point.system_desc.get("tps_utilization"), "float")
            if (
                not isinstance(util, int | float)
                or isinstance(util, bool)
                or not math.isfinite(float(util))
            ):
                continue
            # The comparison walks the points themselves: two points may share a
            # concurrency in a hand-built curve, and neither overwrites the other.
            if abs(float(util) - derived / peak) > TPS_UTILIZATION_ABS_TOL:
                misses.append(point.concurrency)
        return tuple(misses)

    def accuracy_uncovered_regions(self) -> tuple[str, ...]:
        """The §5.3 mandatory regions no accuracy-carrying point lands in.

        §4.3's validation runs must cover the four mandatory regions: a region
        is covered when a point carrying a parsable `accuracy_results.json`
        falls in it. Unlike the curve's own coverage (`uncovered_regions`), a
        dedicated Offline point is not exempt here: the round's checker reads
        every accuracy-carrying point's concurrency regardless of its
        `offline` declaration, and the `margin` band is not a coverage region.

        Returns:
            tuple[str, ...]: One `region` wire value per uncovered mandatory
                region, empty when every one is covered.
        """
        regions = self.regions()
        covered: set[RegionKind] = set()
        for point in self._accuracy_carrying_points():
            kind = regions.covers(point.concurrency)
            if kind is not None:
                covered.add(kind)
        return tuple(kind.value for kind in _MANDATORY_REGIONS if kind not in covered)

    def accuracy_gate_misses(self) -> tuple[str, ...]:
        """The Section 15 gate misses the curve's accuracy results report.

        The round's checker gates each accuracy-carrying run against the
        catalog's targets for the declared `model_name`: the run's per-dataset
        scores aggregate into a sample-weighted mean per metric, compared
        against `accuracy_targets` scaled by its `factor`, and the issued sample
        count (`num_samples x max(n_repeats, 1)` per entry; the client's
        `total_samples` where spelled) must reach the model's `min_queries`. A
        score reported as a fraction (`0..1`) rescales to the percent scale when
        the bound sits above 1, and a metric the run does not report is
        skipped, exactly as the checker skips it. An aggregate holding only the
        client's unnamed `score` is gated against the model's single metric, as
        the checker gates it; its warning that the scalar's identity cannot be
        verified is not a miss here.

        Returns:
            tuple[str, ...]: One line per miss (`c<N>: ...`), empty when every
                gate holds or no catalog entry carries targets to judge.
        """
        catalog = self._catalog()
        model_name = self._model_name()
        if catalog is None or model_name is None:
            return ()
        model = catalog.models.get(model_name)
        if not isinstance(model, Mapping):
            return ()
        bounds = _accuracy_bounds(model.get("accuracy_targets"))
        if not bounds:
            return ()
        raw_min = decode_number(model.get("min_queries"), "float")
        min_samples = (
            float(raw_min)
            if isinstance(raw_min, int | float) and not isinstance(raw_min, bool)
            else None
        )
        misses: list[str] = []
        for point in self._accuracy_carrying_points():
            if point.node is None:
                continue
            value, error = point.node.read("accuracy_results.json")
            if error is not None or not isinstance(value, Mapping):
                continue
            entries = _accuracy_document_entries(value)
            if not entries:
                continue
            issued = [
                count for count in map(_accuracy_issued, entries) if count is not None
            ]
            if min_samples is not None and issued and sum(issued) < min_samples:
                misses.append(
                    f"c{point.concurrency}: {sum(issued):.0f} accuracy "
                    f"samples < {min_samples:.0f}"
                )
            weighted: dict[str, list[tuple[float, float | None]]] = {}
            for entry in entries:
                weight = _accuracy_weight(entry)
                for metric, score in _accuracy_entry_scores(entry).items():
                    weighted.setdefault(metric, []).append((score, weight))
            aggregate = {
                metric: _weighted_mean(pairs) for metric, pairs in weighted.items()
            }
            if list(aggregate) == ["score"] and len(bounds) == 1:
                aggregate = {next(iter(bounds)): aggregate["score"]}
            for metric, (lower, upper) in bounds.items():
                reported = next(
                    (
                        score
                        for name, score in aggregate.items()
                        if name.lower() == metric
                    ),
                    None,
                )
                if reported is None:
                    continue
                if not math.isfinite(float(reported)):
                    misses.append(
                        f"c{point.concurrency}: {metric} reads non-finite ({reported})"
                    )
                    continue
                if 0.0 <= reported <= 1.0 and lower > 1.0:
                    reported *= 100.0
                if reported < lower or (upper is not None and reported > upper):
                    gate = (
                        f"[{lower:.4f}, {upper:.4f}]"
                        if upper is not None
                        else f">= {lower:.4f}"
                    )
                    misses.append(
                        f"c{point.concurrency}: {metric} {reported:.4f} outside {gate}"
                    )
        return tuple(misses)

    def structure_misses(self) -> tuple[str, ...]:
        """The submission tree's Section 8.1 structure misses.

        The checker validates the submission directory itself: `docs/` must
        exist beside `results/`, and `src/` must hold at least one
        implementation directory, each carrying a README (Section 2.2.1). A
        `<system>`-directory input carries no such siblings -- the checker's own
        input is the submission root -- and the check is skipped there.

        Returns:
            tuple[str, ...]: One line per miss, empty when the tree is whole.
        """
        if self.node is None:
            return ()
        root = Path(self.node.root().dir)
        if not (root / "results").is_dir():
            return ()
        misses: list[str] = []
        if not (root / "docs").is_dir():
            misses.append("docs/ is missing beside results/")
        src_dir = root / "src"
        if not src_dir.is_dir():
            misses.append("src/ is missing")
            return tuple(misses)
        implementations = [path for path in sorted(src_dir.iterdir()) if path.is_dir()]
        if not implementations:
            misses.append("src/ holds no implementation directory")
        for implementation in implementations:
            readme = next(
                (
                    path
                    for path in implementation.iterdir()
                    if path.is_file() and path.name.lower() == "readme.md"
                ),
                None,
            )
            if readme is None:
                misses.append(f"src/{implementation.name}/ carries no README.md")
        return tuple(misses)

    def unresolved_shared_paths(self) -> tuple[str, ...]:
        """The `shared_src`/`shared_docs` pointers that do not resolve (§9.1).

        Section 9.1's shared-path resolution: each pointer must name an existing
        directory under the submission root, written relative to it -- never
        absolute, never carrying a `..`. Reported per distinct `field=value`
        pair, as the checker reports it (every point normally names one tree).
        A `<system>`-directory input carries no root-level trees to resolve
        against, and the checker's own input always does.

        Returns:
            tuple[str, ...]: One `"<field>=<value>"` per unresolvable pointer,
                empty when every one resolves.
        """
        if self.node is None:
            return ()
        root = Path(self.node.root().dir)
        if not (root / "results").is_dir():
            return ()
        seen: set[tuple[str, str]] = set()
        unresolved: set[str] = set()
        for point in self.points:
            if point.node is None:
                continue
            config, error = point.node.read("point.yaml")
            if error is not None or not isinstance(config, Mapping):
                continue
            for pointer in ("shared_src", "shared_docs"):
                value = config.get(pointer)
                if not isinstance(value, str) or (pointer, value) in seen:
                    continue
                seen.add((pointer, value))
                if _resolves_shared_path(root, value) is None:
                    unresolved.add(f"{pointer}={value}")
        return tuple(sorted(unresolved))

    def unapproved_drafter_points(self) -> tuple[str, ...]:
        """The points whose speculative decoding is not an approved drafter.

        Section 2.9.4 makes the drafters a curated, published list: a point
        declaring `speculative_decoding` must name an entry approved for its
        benchmark -- by weight checksum, or by target checksum plus the entry's
        own configuration -- and the entry must have been approved at least two
        cohorts before the point's `target_cohort`. The list lives on the
        catalog's model entry (`approved_drafters`), so a model declaring none
        has none, and speculative decoding stays disallowed for it. A point
        whose `target_cohort`, or an entry whose `approved_cohort`, records no
        parseable cohort is not judged here (the checker warns instead).

        Returns:
            tuple[str, ...]: One line per unapproved or too-early drafter, empty
                when every declaration matches an approved entry in time.
        """
        catalog = self._catalog()
        model_name = self._model_name()
        if catalog is None or model_name is None:
            return ()
        model = catalog.models.get(model_name)
        if not isinstance(model, Mapping):
            return ()
        declared_entries = model.get("approved_drafters")
        entries = (
            [entry for entry in declared_entries if isinstance(entry, Mapping)]
            if isinstance(declared_entries, Sequence)
            and not isinstance(declared_entries, str | bytes)
            else []
        )
        misses: list[str] = []
        for point in self.points:
            if point.node is None:
                continue
            config, error = point.node.read("point.yaml")
            if error is not None or not isinstance(config, Mapping):
                continue
            declared = config.get("speculative_decoding")
            if not isinstance(declared, Mapping) or not declared:
                continue
            matched = next(
                (entry for entry in entries if _drafter_entry_matches(entry, declared)),
                None,
            )
            if matched is None:
                misses.append(
                    f"c{point.concurrency}: speculative_decoding matches no "
                    f"approved drafter for {model_name}"
                )
                continue
            target = _advance_cohort(str(config.get("target_cohort", "")), 0)
            earliest = _advance_cohort(
                str(matched.get("approved_cohort", "")),
                _DRAFTER_APPROVAL_LEAD_COHORTS,
            )
            if target is None or earliest is None:
                continue
            if target < earliest:
                misses.append(
                    f"c{point.concurrency}: target cohort {target} precedes the "
                    f"drafter's earliest use {earliest}"
                )
        return tuple(misses)

    def offline_points_missing_accuracy(self) -> tuple[int, ...]:
        """The Offline points carrying no `accuracy_results.json` (§4.3).

        Every Offline point's §4.3 validation result must be shipped: an
        elected C_max point is one of the required results, and a dedicated
        run's result stands beside the four region results
        (`accuracy_uncovered_regions` counts the regions only).

        Returns:
            tuple[int, ...]: The concurrency of each Offline point with no
                parsable accuracy result, empty when all carry one.
        """
        carrying = {id(point) for point in self._accuracy_carrying_points()}
        return tuple(
            point.concurrency
            for point in self.offline_points()
            if id(point) not in carrying
        )

    def _accuracy_carrying_points(self) -> tuple[Point, ...]:
        """The points whose run directory carries a parsable accuracy result.

        The one definition of "carrying": a parsable `accuracy_results.json` whose
        `accuracy_scores` holds at least one entry. An entryless document is no
        accuracy result, so the coverage rows, the gate and the presence row all
        read the same fact.
        """
        carrying: list[Point] = []
        for point in self.points:
            if point.node is None:
                continue
            value, error = point.node.read("accuracy_results.json")
            if error is not None or not isinstance(value, Mapping):
                continue
            if _accuracy_document_entries(value):
                carrying.append(point)
        return tuple(carrying)

    def entryless_accuracy_points(self) -> tuple[int, ...]:
        """The points shipping an `accuracy_results.json` with no dataset entry.

        Section 4.3's accuracy result is the entries: a parsable document whose
        entry set is empty (in either shipped shape) carries no accuracy result
        at all, so it can count neither towards the coverage rows nor the gate.
        Reported per point, so the finding names the run to re-export.

        Returns:
            tuple[int, ...]: The concurrency of each entryless carrier, empty
                when every shipped document holds at least one entry.
        """
        entryless: list[int] = []
        for point in self.points:
            if point.node is None:
                continue
            value, error = point.node.read("accuracy_results.json")
            if error is not None or not isinstance(value, Mapping):
                continue
            if not _accuracy_document_entries(value):
                entryless.append(point.concurrency)
        return tuple(entryless)

    def point_count_min(self) -> int:
        """Endpoints Policy §5.3's minimum point count this pareto needs.

        The four mandatory region points plus the three submitter's-choice
        points. A non-agentic pareto adds the Offline slot - one more distinct
        run (`8`), unless the submitter elects the C_max point as the Offline
        result, which carries no new run (`7`). An agentic pareto runs `7`
        (§5.7: no Offline point applies). Which shape the pareto is keys on
        `agentic()` (the catalog's flag for the declared model), so a
        non-agentic pareto that forgot its Offline declaration is short of
        `8`, not silently re-read as agentic.

        Returns:
            int: `MANDATORY_REGION_COUNT` + `SUBMITTERS_CHOICE_COUNT` (7), plus
                one for a non-agentic pareto with no `elected` declaration (`8`).
        """
        base = MANDATORY_REGION_COUNT + SUBMITTERS_CHOICE_COUNT
        if self.agentic():
            return base
        return base + (0 if any(p.offline == "elected" for p in self.points) else 1)

    def derived(self) -> Mapping[str, Any]:
        """Named derivations for `derived.<name>` row refs, as a lazy mapping.

        The boundary rule for derivations ("named derivations on their subject,
        exposed as fields `Ref` reads"): the keys are `_DERIVATION_METHODS`'
        names - the Offline/point-count/region-coverage/accuracy-coverage
        facts and the catalog-admitted model - a row cannot compute itself -
        each resolved via its own named method at
        first access and memoized (`DerivedFields`). An unlisted name ->
        `KeyError`.

        Returns:
            Mapping[str, Any]: The `DerivedFields` view over this pareto: the
                `_DERIVATION_METHODS` names are its keys, and nothing runs until
                a row names one.
        """
        return DerivedFields(self)


@dataclass(frozen=True)
class RefusedSubmission:
    """One submission the loader could not judge: its path and why.

    Args:
        path (Path): The refused `results/<system>/<model>/` directory, or the
            refused directory itself when no submission sits below it.
        message (str): Why the subtree cannot be judged.
    """

    path: Path
    message: str


def _refuse(node: Node, message: str) -> None:
    """Record `message` on `node`: its subtree is unjudgeable from here."""
    node.refusals = (*node.refusals, message)


def _refused_submissions(node: Node) -> list[RefusedSubmission]:
    """`node`'s refusals as per-submission records, one per pareto below it.

    A refused node with no pareto below it (a system directory holding no
    model directory) keeps one record of its own: the refusal is real and must
    be reported even though no submission can carry it.
    """
    paretos = [child for child in node.children if child.kind is RuleScopes.PARETO]
    if not paretos:
        return [RefusedSubmission(node.dir, refusal) for refusal in node.refusals]
    return [
        RefusedSubmission(pareto.dir, refusal)
        for pareto in paretos
        for refusal in node.refusals
    ]


@dataclass(frozen=True)
class Submission:
    """One uploaded tree: its paretos, and the nodes rows resolve facts from.

    `Submission.from_directory(root, ruleset=...)` walks
    `results/<system>/<model>/r<N>/` and builds one `Pareto` per
    `<system>/<model>` -- each run directory carrying its own
    `system_desc.json` (§8.2.1), `point.yaml` (§8.3) and
    `result_summary.json` (§8.1). `pareto_at(node)` locates the Pareto a tree
    node belongs to (nodes above every pareto have none: there is no
    single-pareto `derived`/`regions` view of a multi-pareto upload).

    Args:
        tree (Node): The uploaded tree. Its `<system>` children hold the
            `<model>` nodes the paretos hang from.
        paretos (Sequence[Pareto]): One `Pareto` per `<system>/<model>` of the
            tree, in tree order.
        refused (tuple[RefusedSubmission, ...]): One record per submission the
            loader could not judge, so a pool may mix judged and refused
            submissions. (Default: ())
    """

    tree: Node
    paretos: Sequence[Pareto]
    refused: tuple[RefusedSubmission, ...] = ()
    points: tuple[Point, ...] = field(init=False, repr=False, compare=False)
    # Pareto node id -> its `Pareto`: the one lookup `pareto_at` needs. `id()`
    # keys are safe here: nodes compare by identity and the tree outlives this.
    _by_node: dict[int, Pareto] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paretos", tuple(self.paretos))
        object.__setattr__(
            self,
            "points",
            tuple(point for pareto in self.paretos for point in pareto.points),
        )
        object.__setattr__(
            self, "_by_node", {id(pareto.node): pareto for pareto in self.paretos}
        )

    @classmethod
    def from_directory(cls, root: Path, ruleset: Catalog | None = None) -> Submission:
        """Build one from its uploaded tree rather than by hand.

        A problem that keeps one submission (one `<system>/<model>`) from being
        judged is carried as a `RefusedSubmission` instead of raised, so the
        other submissions of a pool still get their findings. A problem of the
        input itself -- an unwalkable root, a pool listing no system, or an
        entry directly under the root that no kind can take -- stays an error.
        """
        tree = SubmissionNode.from_directory(root, ruleset=ruleset)
        if tree.refusals:
            raise ValueError("; ".join(tree.refusals))
        paretos: list[Pareto] = []
        refused: list[RefusedSubmission] = []
        for system in tree.children:
            if system.refusals:
                refused.extend(_refused_submissions(system))
                continue
            if not system.children:
                _refuse(system, f"{system.dir} has no model directories to check")
                refused.extend(_refused_submissions(system))
                continue
            for pareto_node in system.children:
                if pareto_node.refusals:
                    refused.extend(_refused_submissions(pareto_node))
                    continue
                if not pareto_node.children:
                    _refuse(
                        pareto_node,
                        f"{pareto_node.dir} has no run directories (r<N>) to check",
                    )
                    refused.extend(_refused_submissions(pareto_node))
                    continue
                try:
                    paretos.append(Pareto.from_nodes(pareto_node))
                except ValueError as exc:
                    _refuse(pareto_node, str(exc))
                    refused.extend(_refused_submissions(pareto_node))
        if not paretos and not refused:
            raise ValueError(
                f"{root} holds no paretos: a pool root expects "
                "results/<system>/<model>/r<N>/, a system directory "
                "<model>/r<N>/"
            )
        return cls(tree, tuple(paretos), tuple(refused))

    def pareto_at(self, node: Node) -> Pareto | None:
        """The Pareto `node` belongs to. `None` for nodes above every pareto."""
        probe: Node | None = node
        while probe is not None and probe.kind is not RuleScopes.PARETO:
            probe = probe.parent
        if probe is None:
            return None
        return self._by_node.get(id(probe))


def _shared_facts(points: Sequence[Point], where: Path) -> Mapping[str, Any]:
    """The declared facts one pareto's points must agree on (§8.3/§5.4).

    Agreement is presence plus typed value: an omitted key, an explicit `null`
    and a value of another type are three different declarations, so a later
    run cannot smuggle an invalid override (e.g. `c_min_override: null`) past a
    comparison against the first run's spelling.
    """
    if not points:
        raise ValueError(f"{where} has no run directories (r<N>) to check")
    first = points[0].system_desc
    for point in points[1:]:
        for key in ("max_supported_concurrency", "c_min_override"):
            if not _same_declaration(first, point.system_desc, key):
                raise ValueError(
                    f"every point of {where} must declare the same {key} "
                    "(Endpoints Policy §8.3; §5.4, C_min override)"
                )
    return first


def _same_declaration(
    first: Mapping[str, Any], other: Mapping[str, Any], key: str
) -> bool:
    """Whether both descriptors declare `key` identically: presence and typed value."""
    if (key in first) != (key in other):
        return False
    if key not in first:
        return True
    first_value, other_value = first[key], other[key]
    # Same type exactly, not `isinstance`: `True` must not agree with `1`, nor
    # `16.0` with `16` -- the conflations this comparison exists to refuse.
    if type(first_value) is not type(other_value):  # noqa: E721
        return False
    return first_value == other_value


def _same_value(first: Any, other: Any) -> bool:
    """Whether two declared values read as the same declaration.

    A number and its stringified spelling are the same declaration: every
    numeric fact decodes through `decode_number`, so a curve mixing `8192` and
    `"8192"` describes one system. `True` never agrees with `1`, and a
    non-numeric value still compares by exact type.

    Args:
        first (Any): One declared value, as the artifact spells it.
        other (Any): The value another point spells for the same key.
    Returns:
        bool: Whether both read as the same declaration.
    """
    if isinstance(first, bool) or isinstance(other, bool):
        return isinstance(first, bool) and isinstance(other, bool) and first == other
    decoded_first = decode_number(first, "float")
    decoded_other = decode_number(other, "float")
    numbers = (decoded_first, decoded_other)
    if all(
        isinstance(number, int | float) and not isinstance(number, bool)
        for number in numbers
    ):
        return float(decoded_first) == float(decoded_other)
    return type(decoded_first) is type(decoded_other) and decoded_first == decoded_other  # noqa: E721


class DerivedFields(Mapping[str, Any]):
    """Lazy `derived.<name>` mapping: a derivation runs only when a row names it.

    `Ref.resolve` walks this like any namespace. `view["point_cap"]` ->
    `Pareto.point_cap()`'s value, memoized. `view["nope"]` -> `KeyError`, so
    `Ref.resolve` reports the field absent. Membership asks the known-names

    tables only and never runs a derivation, so one derivation's fail-loud
    `ValueError` reaches exactly the row that named it.

    Args:
        pareto (Pareto): The pareto whose named derivations this mapping exposes.
    """

    __slots__ = ("_pareto", "_cache")

    def __init__(self, pareto: Pareto) -> None:
        self._pareto = pareto
        self._cache: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        """Value of `derived.<name>`: runs its derivation once, memoized."""
        if name in self._cache:
            return self._cache[name]
        try:
            method = _DERIVATION_METHODS[name]
        except KeyError:
            raise KeyError(name) from None
        value = method(self._pareto)
        self._cache[name] = value
        return value

    def __contains__(self, name: object) -> bool:
        """Whether `name` is a known derivation name. Never runs one."""
        return name in _DERIVATION_METHODS

    def __iter__(self) -> Iterator[str]:
        """Every known derivation name (the `derived.<name>` key set)."""
        return iter(_DERIVATION_METHODS)

    def __len__(self) -> int:
        return len(_DERIVATION_METHODS)


_DERIVATION_METHODS: dict[str, Callable[[Pareto], Any]] = {
    "offline_ordering": Pareto.offline_ordering,
    "point_cap": Pareto.point_cap,
    "point_count": Pareto.point_count,
    "point_count_min": Pareto.point_count_min,
    "accuracy_uncovered_regions": Pareto.accuracy_uncovered_regions,
    "offline_points_missing_accuracy": Pareto.offline_points_missing_accuracy,
    "accounting_mismatch_points": Pareto.accounting_mismatch_points,
    "system_tps_mismatch_points": Pareto.system_tps_mismatch_points,
    "misplaced_region_points": Pareto.misplaced_region_points,
    "tps_utilization_mismatch_points": Pareto.tps_utilization_mismatch_points,
    "accuracy_gate_misses": Pareto.accuracy_gate_misses,
    "structure_misses": Pareto.structure_misses,
    "unresolved_shared_paths": Pareto.unresolved_shared_paths,
    "unapproved_drafter_points": Pareto.unapproved_drafter_points,
    "entryless_accuracy_points": Pareto.entryless_accuracy_points,
    "uncovered_regions": Pareto.uncovered_regions,
    "offline_points": Pareto.offline_points,
    "agentic": Pareto.agentic,
    "model_name_admitted": Pareto.model_name_admitted,
    "conflicting_load_patterns": Pareto.conflicting_load_patterns,
    "conflicting_model_names": Pareto.conflicting_model_names,
    "conflicting_datasets": Pareto.conflicting_datasets,
    "conflicting_seed_sets": Pareto.conflicting_seed_sets,
    "inconsistent_system_desc_keys": Pareto.inconsistent_system_desc_keys,
    "out_of_range_points": lambda pareto: pareto.out_of_range_points(pareto.regions()),
}

# The `derived.<name>` names a row may name (single owner: the table above).
DERIVATION_NAMES: tuple[str, ...] = tuple(_DERIVATION_METHODS)
