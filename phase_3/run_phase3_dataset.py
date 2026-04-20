"""Phase-3 dataset generator runner (WSL ns-3).

This script:
- Copies phase_3/phase3_eval_scenarios.cc into your ns-3 scratch/ folder in WSL
- Runs the full scenario/pattern/seed matrix
- Saves outputs ONLY into this repo under phase_3/test_dataset

Outputs per run:
- <prefix>_tick.csv
- <prefix>_events.csv
- <prefix>_summary.json

Typical usage (Windows PowerShell):
  python phase_3/run_phase3_dataset.py --seedStart 11 --seedEnd 15

Quick usage (smaller + faster to evaluate):
    python phase_3/run_phase3_dataset.py --preset quick --clean

Notes:
- Requires WSL and an ns-3 checkout that supports: ./ns3 run "scratch/<prog> ..."
- Defaults assume distro=Ubuntu and ns-3 root at ~/ns-3-dev.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class WslConfig:
    distro: str
    ns3_root: str


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCE = WORKSPACE_ROOT / "phase_3" / "phase3_eval_scenarios.cc"
LOCAL_OUT_DIR = WORKSPACE_ROOT / "phase_3" / "test_dataset"


PRESETS = {
    # Original full evaluation matrix.
    "full": {
        "scenarioIds": "8,9,10",
        "patterns": "A,B,C",
        "seedStart": 11,
        "seedEnd": 15,
        "seeds": "",
        "ueCount": 20,
        "duration": 0,
    },
    # Fast-but-representative matrix:
    # - scenarios 8/9/10 (ping-pong, non-stationary interference, coverage hole)
    # - patterns B/C (dynamic + stop-go variants)
    # - two seeds for basic robustness
    # - reduced UE count and fixed duration (covers scenario 9 burst + scenario 10 U-turn)
    "quick": {
        "scenarioIds": "8,9,10",
        "patterns": "B,C",
        "seedStart": 11,
        "seedEnd": 12,
        "seeds": "",
        "ueCount": 8,
        "duration": 300,
    },
}


def _windows_path_to_wsl(path: Path) -> str:
    """Convert an absolute Windows path like E:\foo\bar -> /mnt/e/foo/bar."""
    resolved = path.resolve()
    drive = resolved.drive
    if not drive or not drive.endswith(":"):
        raise ValueError(f"Unsupported path (expected drive letter): {resolved}")

    drive_letter = drive[0].lower()
    # parts[0] is like 'E:\\'
    parts = resolved.parts
    rel_parts = parts[1:]
    rel_posix = "/".join(rel_parts)
    return f"/mnt/{drive_letter}/{rel_posix}"


def _run_wsl_bash(wsl: WslConfig, bash_cmd: str) -> subprocess.CompletedProcess:
    """Run a bash command inside WSL."""
    cmd = ["wsl", "-d", wsl.distro, "-e", "bash", "-c", bash_cmd]
    return subprocess.run(cmd, capture_output=True, text=True)


def _parse_int_list(csv: str) -> List[int]:
    items: List[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _parse_pattern_list(csv: str) -> List[str]:
    items: List[str] = []
    for part in csv.split(","):
        part = part.strip().upper()
        if not part:
            continue
        if part not in {"A", "B", "C"}:
            raise ValueError(f"Invalid pattern '{part}'. Use A,B,C.")
        items.append(part)
    return items


def _expand_seeds(seed_start: int, seed_end: int, explicit: Sequence[int] | None) -> List[int]:
    if explicit and len(explicit) > 0:
        return [int(s) for s in explicit]
    if seed_end < seed_start:
        raise ValueError("seedEnd must be >= seedStart")
    return list(range(int(seed_start), int(seed_end) + 1))


def _ensure_local_dirs() -> None:
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_output_dir() -> None:
    """Delete generated Phase-3 outputs in phase_3/test_dataset."""
    if not LOCAL_OUT_DIR.exists():
        return

    removed = 0
    for pattern in ("*_tick.csv", "*_events.csv", "*_summary.json"):
        for p in LOCAL_OUT_DIR.glob(pattern):
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass

    print(f"Cleaned {removed} files from {LOCAL_OUT_DIR}")


def deploy_cpp_to_ns3_scratch(wsl: WslConfig) -> None:
    if not CPP_SOURCE.exists():
        raise FileNotFoundError(f"Missing C++ generator: {CPP_SOURCE}")

    ws_wsl = _windows_path_to_wsl(WORKSPACE_ROOT)

    bash = (
        "set -euo pipefail; "
        f"cd {wsl.ns3_root}; "
        f"cp {ws_wsl}/phase_3/phase3_eval_scenarios.cc scratch/phase3_eval_scenarios.cc"
    )

    result = _run_wsl_bash(wsl, bash)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to deploy C++ file into ns-3 scratch.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def run_one(wsl: WslConfig, scenario_id: int, pattern: str, seed: int, ue_count: int, duration: int, ttt_ms: int, hys_db: float) -> Path:
    _ensure_local_dirs()

    run_prefix = f"s{scenario_id}_p{pattern}_seed{seed}"
    local_prefix = LOCAL_OUT_DIR / run_prefix

    out_prefix_wsl = _windows_path_to_wsl(local_prefix)

    # Ensure the output dir exists from inside WSL (in case WSL can't see a just-created Windows dir yet).
    out_dir_wsl = _windows_path_to_wsl(LOCAL_OUT_DIR)

    ns3_args = (
        "scratch/phase3_eval_scenarios "
        f"--scenarioId={scenario_id} "
        f"--pattern={pattern} "
        f"--seed={seed} "
        f"--ueCount={ue_count} "
        f"--duration={duration} "
        f"--tttMs={ttt_ms} "
        f"--hysDb={hys_db} "
        f"--outputPrefix={out_prefix_wsl}"
    )

    bash = (
        "set -euo pipefail; "
        f"mkdir -p {out_dir_wsl}; "
        f"cd {wsl.ns3_root}; "
        f"./ns3 run '{ns3_args}'"
    )

    result = _run_wsl_bash(wsl, bash)
    if result.returncode != 0:
        raise RuntimeError(
            f"ns-3 run failed for {run_prefix}.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    tick = Path(str(local_prefix) + "_tick.csv")
    events = Path(str(local_prefix) + "_events.csv")
    summary = Path(str(local_prefix) + "_summary.json")

    missing = [p for p in (tick, events, summary) if not p.exists()]
    if missing:
        raise RuntimeError(
            "ns-3 run completed but expected outputs are missing in phase_3/test_dataset:\n"
            + "\n".join(str(p) for p in missing)
        )

    return local_prefix


def generate_matrix(
    wsl: WslConfig,
    scenario_ids: Sequence[int],
    patterns: Sequence[str],
    seeds: Sequence[int],
    ue_count: int,
    duration: int,
    ttt_ms: int,
    hys_db: float,
) -> None:
    deploy_cpp_to_ns3_scratch(wsl)

    total = len(scenario_ids) * len(patterns) * len(seeds)
    done = 0

    for scenario_id in scenario_ids:
        for pattern in patterns:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] scenario={scenario_id} pattern={pattern} seed={seed}")
                run_one(
                    wsl=wsl,
                    scenario_id=int(scenario_id),
                    pattern=str(pattern),
                    seed=int(seed),
                    ue_count=int(ue_count),
                    duration=int(duration),
                    ttt_ms=int(ttt_ms),
                    hys_db=float(hys_db),
                )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Phase-3 unseen ns-3 datasets into phase_3/test_dataset")

    p.add_argument(
        "--preset",
        default="full",
        choices=sorted(PRESETS.keys()),
        help="Dataset preset to use (default: full)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing generated outputs in phase_3/test_dataset before regenerating",
    )

    p.add_argument("--scenarioIds", default="8,9,10", help="Comma-separated scenario IDs (default: 8,9,10)")
    p.add_argument("--patterns", default="A,B,C", help="Comma-separated patterns (default: A,B,C)")

    p.add_argument("--seedStart", type=int, default=11, help="Seed range start (inclusive)")
    p.add_argument("--seedEnd", type=int, default=15, help="Seed range end (inclusive)")
    p.add_argument("--seeds", default="", help="Optional explicit comma-separated seeds (overrides seedStart/seedEnd)")

    p.add_argument("--ueCount", type=int, default=20, help="Number of UEs per run")
    p.add_argument("--duration", type=int, default=0, help="Duration seconds (0 uses pattern default)")
    p.add_argument("--tttMs", type=int, default=160, help="TTT in ms used by generator HO engine")
    p.add_argument("--hysDb", type=float, default=3.0, help="Hysteresis in dB used by generator HO engine")

    p.add_argument("--wslDistro", default="Ubuntu", help="WSL distro name (default: Ubuntu)")
    p.add_argument("--wslNs3Root", default="~/ns-3-dev", help="ns-3 root path inside WSL (default: ~/ns-3-dev)")

    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    preset = PRESETS.get(str(args.preset))
    if not preset:
        raise ValueError(f"Unknown preset: {args.preset}")

    # Apply preset (explicit CLI args can still override by passing them after --preset).
    # We keep this simple: preset becomes the baseline, then argparse-provided values win.
    # (Users who want pure preset behavior can just pass --preset ... with no overrides.)
    if args.scenarioIds == _build_arg_parser().get_default("scenarioIds"):
        args.scenarioIds = preset["scenarioIds"]
    if args.patterns == _build_arg_parser().get_default("patterns"):
        args.patterns = preset["patterns"]
    if args.seedStart == _build_arg_parser().get_default("seedStart"):
        args.seedStart = preset["seedStart"]
    if args.seedEnd == _build_arg_parser().get_default("seedEnd"):
        args.seedEnd = preset["seedEnd"]
    if str(args.seeds) == _build_arg_parser().get_default("seeds"):
        args.seeds = preset["seeds"]
    if args.ueCount == _build_arg_parser().get_default("ueCount"):
        args.ueCount = preset["ueCount"]
    if args.duration == _build_arg_parser().get_default("duration"):
        args.duration = preset["duration"]

    wsl = WslConfig(distro=str(args.wslDistro), ns3_root=str(args.wslNs3Root))

    scenario_ids = _parse_int_list(str(args.scenarioIds))
    patterns = _parse_pattern_list(str(args.patterns))

    explicit_seeds = _parse_int_list(str(args.seeds)) if str(args.seeds).strip() else None
    seeds = _expand_seeds(int(args.seedStart), int(args.seedEnd), explicit_seeds)

    if any(sid not in {8, 9, 10} for sid in scenario_ids):
        raise ValueError("Phase-3 generator supports scenarioIds {8,9,10}.")

    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

    if bool(args.clean):
        _clean_output_dir()

    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Output dir: {LOCAL_OUT_DIR}")
    print(f"WSL: distro={wsl.distro} ns3_root={wsl.ns3_root}")
    print(f"Preset: {args.preset}")
    print(f"Matrix: scenarios={scenario_ids} patterns={patterns} seeds={seeds} ueCount={int(args.ueCount)} duration={int(args.duration)}")

    generate_matrix(
        wsl=wsl,
        scenario_ids=scenario_ids,
        patterns=patterns,
        seeds=seeds,
        ue_count=int(args.ueCount),
        duration=int(args.duration),
        ttt_ms=int(args.tttMs),
        hys_db=float(args.hysDb),
    )

    print("Done. Outputs are in phase_3/test_dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
