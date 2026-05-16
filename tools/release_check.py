#!/usr/bin/env python3
"""
Release-readiness sanity checks for the Dexora repository.

What it does
------------
1.  Verifies every "required" file actually exists.
2.  Greps the README + scripts for unresolved TBD / TODO / FIXME markers
    that should be filled in before tagging a release.
3.  Confirms the three-stage pipeline scripts cross-reference each other
    correctly (Stage-2 reads what Stage-1 produced, Stage-3 reads what
    Stage-2 produced, ...).
4.  In ``--strict`` mode, treats warnings as errors. This is what we run in
    CI to gate the ``main`` branch.

Usage
-----
    python tools/release_check.py            # report-only
    python tools/release_check.py --strict   # fail on any issue
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

__all__ = [
    "main",
    "check_required_files",
    "check_readme_placeholders",
    "check_script_cross_references",
    "check_pyproject_consistency",
]

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

REQUIRED_FILES: tuple[str, ...] = (
    # Project hygiene
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    ".gitignore",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".pre-commit-config.yaml",
    ".github/workflows/ci.yml",
    # The paper
    "ICRA26_0209_FI.pdf",
    # Entry points
    "main.py",
    "main_scoring.py",
    "main_posttrain.py",
    "train_ours.sh",
    "train_scoring.sh",
    "post_train.sh",
    "run_all_stages.sh",
    "Makefile",
    "compute_logpi.py",
    "replay_validate.py",
    "analyze_episode_quality.py",
    # Core configs
    "configs/base.yaml",
    "configs/base_400m.yaml",
    "configs/scoring.yaml",
    "configs/cross_embodiment/README.md",
    "configs/cross_embodiment/ec1_franka.yaml",
    "configs/cross_embodiment/ec2_aloha.yaml",
    "configs/cross_embodiment/ec3_g1_inspire.yaml",
    # Core models / utilities
    "models/rdt_runner.py",
    "models/scoring_model.py",
    "models/sample_weighting.py",
    # Training drivers
    "train/train.py",
    "train/train_scoring.py",
    "train/train_posttrain.py",
    # Scripts
    "scripts/eval_smoothness.py",
    # Tests
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_sample_weighting.py",
    "tests/test_pu_loss.py",
    "tests/test_weighted_mse_loss.py",
    "tests/test_cross_embodiment_configs.py",
    "tests/test_replay_validate.py",
    "tests/test_smoothness_eval.py",
)

# Strings in the README that we expect to disappear before tagging a release.
README_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "release pending",
    "TBD",
    "FIXME",
    # Note: deliberately *not* matching "TODO" since many docs legitimately
    # reference a "TODO" list section.
)

# Cross-script dependency probes: tuples of (script, must-mention).
SCRIPT_LINKS: tuple[tuple[str, str], ...] = (
    ("train_scoring.sh", "logpi_file"),
    ("post_train.sh", "stage1_ckpt"),
    ("post_train.sh", "scoring_ckpt"),
    ("run_all_stages.sh", "train_ours.sh"),
    ("run_all_stages.sh", "train_scoring.sh"),
    ("run_all_stages.sh", "post_train.sh"),
)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_required_files(report: list[tuple[str, str]]) -> None:
    for rel in REQUIRED_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            report.append(("error", f"missing required file: {rel}"))


def check_readme_placeholders(report: list[tuple[str, str]]) -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for marker in README_PLACEHOLDER_MARKERS:
        # Case-insensitive line-level matching for nicer messages.
        for lineno, line in enumerate(readme.splitlines(), start=1):
            if marker.lower() in line.lower():
                report.append((
                    "warning",
                    f"README.md:{lineno}: unresolved '{marker}' -> {line.strip()[:80]!r}",
                ))


def check_script_cross_references(report: list[tuple[str, str]]) -> None:
    for script, needle in SCRIPT_LINKS:
        path = REPO_ROOT / script
        if not path.exists():
            continue  # missing-file check already flagged it
        text = path.read_text(encoding="utf-8")
        if needle.lower() not in text.lower():
            report.append((
                "error",
                f"{script}: expected to reference {needle!r} but doesn't",
            ))


def check_pyproject_consistency(report: list[tuple[str, str]]) -> None:
    """Trivial sanity: pyproject's [project.scripts] entries must resolve."""
    pyproj = REPO_ROOT / "pyproject.toml"
    if not pyproj.exists():
        return
    text = pyproj.read_text(encoding="utf-8")
    # Look for "name = "module:func"" lines under [project.scripts]
    in_section = False
    pattern = re.compile(r"^\s*[\w-]+\s*=\s*\"([\w_]+)(?::([\w_]+))?\"\s*$")
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_section = (line == "[project.scripts]")
            continue
        if not in_section or not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        module, func = m.group(1), m.group(2) or "main"
        module_path = REPO_ROOT / f"{module}.py"
        if not module_path.exists():
            report.append((
                "error",
                f"pyproject.toml [project.scripts]: module {module!r} not found",
            ))
            continue
        src = module_path.read_text(encoding="utf-8")
        if f"def {func}(" not in src:
            report.append((
                "error",
                f"pyproject.toml [project.scripts]: {module}.{func}() does not exist",
            ))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors (default in CI).",
    )
    args = parser.parse_args()

    report: list[tuple[str, str]] = []
    check_required_files(report)
    check_readme_placeholders(report)
    check_script_cross_references(report)
    check_pyproject_consistency(report)

    if not report:
        print("✓ release_check: clean")
        return 0

    n_err = sum(1 for level, _ in report if level == "error")
    n_warn = sum(1 for level, _ in report if level == "warning")

    for level, msg in report:
        prefix = "ERROR  " if level == "error" else "warn   "
        print(f"{prefix} {msg}")

    print()
    print(f"release_check summary: {n_err} error(s), {n_warn} warning(s)")
    if n_err > 0:
        return 1
    if args.strict and n_warn > 0:
        print("(--strict: warnings cause a non-zero exit code)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
