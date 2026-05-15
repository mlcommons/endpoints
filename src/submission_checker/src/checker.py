"""Submission checker — orchestrates §9.1 automated compliance checks.

Loading and structural validation live here; all rule logic lives in Pydantic
model validators on PointConfig, PointResult, and ModelContext.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .loader import (
    load_accuracy_result,
    load_point_config,
    load_result_summary,
    load_system_description,
)
from .models import (
    CheckResult,
    ModelContext,
    PointConfig,
    PointResult,
    PointSummary,
    Regions,
    Report,
    RuntimeSettings,
    Severity,
    SystemDescription,
    compute_regions,
)
from .models import err as _err
from .models import ok as _ok
from .models import warn as _warn
from .structure import ModelDir, SrcDir, SubmissionDir, SystemPareto

if TYPE_CHECKING:
    from pathlib import Path


class SubmissionChecker:
    """Validates an MLPerf Endpoints submission directory against §9.1 rules.

    The *submission_path* should be the submitting organisation's root directory,
    which must contain ``systems/`` and ``pareto/`` subdirectories as specified
    in §8.1.

    Args:
        submission_path: Root directory of the submission to validate.

    Example::

        checker = SubmissionChecker(Path("/submissions/acme_corp"))
        report = checker.run()
        for err in report.errors:
            print(err.rule, err.message)
    """

    def __init__(self, submission_path: Path) -> None:
        self.submission_path = submission_path

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Report:
        """Run all §9.1 automated checks and return an aggregated
        :class:`~submission_checker.models.Report`.

        Returns:
            A :class:`~submission_checker.models.Report` with every
            :class:`~submission_checker.models.CheckResult` produced.
        """
        report = Report(submission_path=self.submission_path)

        if not self.submission_path.exists():
            report.results.append(
                _err(
                    "path-exists",
                    f"Submission path does not exist: {self.submission_path}",
                    self.submission_path,
                    "#1",
                )
            )
            return report
        report.results.append(
            _ok("path-exists", "Submission path exists", self.submission_path, "#1")
        )

        submission_dir = SubmissionDir(root=self.submission_path)
        report.results.extend(submission_dir._check_results)
        if any(r.severity == Severity.ERROR for r in submission_dir._check_results):
            return report

        systems_dir = submission_dir.systems_dir
        pareto_dir = submission_dir.pareto_dir

        system_jsons = sorted(systems_dir.glob("*.json"))
        if not system_jsons:
            report.results.append(
                _err(
                    "system-description-present",
                    "No *.json files found in systems/",
                    systems_dir,
                    "#1",
                )
            )
            return report

        for system_json in system_jsons:
            report.results.extend(self._check_system(system_json, pareto_dir))

        return report

    # ------------------------------------------------------------------
    # Per-system orchestration
    # ------------------------------------------------------------------

    def _check_system(self, system_json: Path, pareto_dir: Path) -> list[CheckResult]:
        results: list[CheckResult] = []
        system_id = system_json.stem

        system_desc, load_err = load_system_description(system_json)
        if load_err:
            results.append(_err("system-description-valid", load_err, system_json, "#1"))
            return results
        assert system_desc is not None
        results.append(
            _ok(
                "system-description-valid",
                f"System description valid: {system_id}",
                system_json,
                "#1",
            )
        )

        M = system_desc.max_supported_concurrency
        results.append(
            _ok("max-concurrency-declared", f"max_supported_concurrency = {M}", system_json, "#7")
        )

        try:
            regions = compute_regions(M)
        except ValueError as exc:
            results.append(_err("region-computation", str(exc), system_json, "#7"))
            return results

        src = SrcDir(root=self.submission_path, division=system_desc.division)
        results.extend(src._check_results)

        system_pareto = SystemPareto(pareto_dir=pareto_dir, system_id=system_id)
        results.extend(system_pareto._check_results)
        if any(r.severity == Severity.ERROR for r in system_pareto._check_results):
            return results
        system_pareto_dir = system_pareto.system_dir
        model_dirs = [d for d in sorted(system_pareto_dir.iterdir()) if d.is_dir()]
        if not model_dirs:
            results.append(
                _err(
                    "benchmark-model-dir",
                    f"No benchmark-model directories in pareto/{system_id}/",
                    system_pareto_dir,
                    "#1",
                )
            )
            return results

        for model_dir in model_dirs:
            results.extend(self._check_model(system_id, system_desc, regions, model_dir))

        return results

    # ------------------------------------------------------------------
    # Per benchmark-model orchestration
    # ------------------------------------------------------------------

    def _check_model(
        self,
        system_id: str,
        system_desc: SystemDescription,
        regions: Regions,
        model_dir: Path,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        benchmark_model = model_dir.name

        model_structure = ModelDir(
            root=model_dir, system_id=system_id, benchmark_model=benchmark_model
        )
        results.extend(model_structure._check_results)
        if any(r.severity == Severity.ERROR for r in model_structure._check_results):
            return results

        points_dir = model_structure.points_dir
        results_dir = model_structure.results_dir
        accuracy_dir = model_structure.accuracy_dir

        point_yamls = sorted(points_dir.glob("point_*.yaml"))
        if not point_yamls:
            results.append(
                _err(
                    "measurement-points-present",
                    f"No point_*.yaml files in {points_dir.relative_to(self.submission_path)}",
                    points_dir,
                    "#1",
                )
            )
            return results

        valid_points: list[tuple[Path, PointConfig]] = []
        loaded_points: list[tuple[PointConfig, PointSummary]] = []

        for yaml_path in point_yamls:
            config, config_results = load_point_config(
                yaml_path, context={"regions": regions, "yaml_path": yaml_path}
            )
            results.extend(config_results)
            if config is None:
                continue

            # filename-concurrency consistency warning
            try:
                fname_concurrency = int(yaml_path.stem.split("_")[1])
                if fname_concurrency != config.concurrency:
                    results.append(
                        _warn(
                            "point-filename-concurrency",
                            f"{yaml_path.name}: filename concurrency {fname_concurrency}"
                            f" ≠ declared {config.concurrency}",
                            yaml_path,
                            "#1",
                        )
                    )
            except (IndexError, ValueError):
                pass

            valid_points.append((yaml_path, config))

            point_result_dir = results_dir / f"point_{config.concurrency}"
            summary_path = point_result_dir / "mlperf_endpoints_log_summary.json"
            detail_path = point_result_dir / "mlperf_endpoints_log_detail.json"

            if not summary_path.exists():
                results.append(
                    _err(
                        "result-file-present",
                        f"Missing result log for point_{config.concurrency}:"
                        f" {summary_path.relative_to(self.submission_path)}",
                        summary_path,
                        "#1",
                    )
                )
                continue

            if not detail_path.exists():
                results.append(
                    _err(
                        "result-detail-present",
                        f"Missing detail log for point_{config.concurrency}:"
                        f" {detail_path.relative_to(self.submission_path)}",
                        detail_path,
                        "#1",
                    )
                )

            summary, load_err = load_result_summary(summary_path)
            if load_err:
                results.append(_err("result-file-valid", load_err, summary_path, "#1"))
                continue

            # PointResult validates point-duration and metric-consistency
            point_result = PointResult.model_validate(
                {"config": config, "summary": summary, "yaml_path": yaml_path},
                context={"regions": regions, "summary_path": summary_path},
            )
            results.extend(point_result._check_results)
            loaded_points.append((config, summary))

        # Load accuracy
        accuracy_result = None
        txt_path = accuracy_dir / "accuracy.txt"
        if not txt_path.exists():
            results.append(_err("accuracy-file", "Missing accuracy/accuracy.txt", txt_path, "#15"))
        json_path = accuracy_dir / "accuracy_result.json"
        if not json_path.exists():
            results.append(
                _err("accuracy-file", "Missing accuracy/accuracy_result.json", json_path, "#15")
            )
        else:
            accuracy_result, acc_results = load_accuracy_result(json_path)
            results.extend(acc_results)
            if any(r.severity == Severity.ERROR for r in acc_results):
                accuracy_result = None

        # ModelContext validates point-count, regional-coverage, config-consistency, accuracy-gate
        model_ctx = ModelContext(
            system_id=system_id,
            system_desc=system_desc,
            model_dir=model_dir,
            regions=regions,
            points_dir=points_dir,
            accuracy_dir=accuracy_dir,
            all_point_count=len(point_yamls),
            valid_points=valid_points,
            loaded_points=loaded_points,
            accuracy_result=accuracy_result,
        )
        results.extend(model_ctx._check_results)

        return results
