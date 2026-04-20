"""Training dataset generator runner (WSL ns-3).

This script:
- Copies dataset.cpp into your ns-3 scratch/ folder in WSL (as md_scenarios.cc)
- Runs the full scenario/pattern/seed matrix for Phase-2 PPO training
- Saves outputs ONLY into this repo under dataset/

Outputs per run:
- <prefix>_tick.csv
- <prefix>_events.csv
- <prefix>_summary.json

Typical usage (Windows PowerShell):
  python run_training_dataset.py --preset full --clean

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


WORKSPACE_ROOT = Path(__file__).resolve().parent
CPP_SOURCE = WORKSPACE_ROOT / "dataset.cpp"
LOCAL_OUT_DIR = WORKSPACE_ROOT / "dataset"


PRESETS = {
    # Phase-2 PPO training matrix (leakage-safe with seeds 1..5).
    "full": {
        "scenarioIds": "1,2,3,4,5,6,7",
        "patterns": "A,B,C",
        "seedStart": 1,
        "seedEnd": 5,
        "seeds": "",
        "ueCount": 20,
        "duration": 0,
        "tttMs": 160,
        "hysDb": 3.0,
    },
}


def _windows_path_to_wsl(path: Path) -> str:
    """Convert an absolute Windows path like E:\foo\bar -> /mnt/e/foo/bar."""
    resolved = path.resolve()
    drive = resolved.drive
    if not drive or not drive.endswith(":"):
        raise ValueError(f"Unsupported path (expected drive letter): {resolved}")

    drive_letter = drive[0].lower()
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
    """Delete generated outputs in dataset/."""
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
    """Copy dataset.cpp into ns-3 scratch as md_scenarios.cc."""
    if not CPP_SOURCE.exists():
        raise FileNotFoundError(f"Missing C++ generator: {CPP_SOURCE}")

    ws_wsl = _windows_path_to_wsl(WORKSPACE_ROOT)

    bash = (
        "set -euo pipefail; "
        f"cd {wsl.ns3_root}; "
        f"cp {ws_wsl}/dataset.cpp scratch/md_scenarios.cc"
    )

    result = _run_wsl_bash(wsl, bash)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to deploy dataset.cpp into ns-3 scratch.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def run_one(
    wsl: WslConfig,
    scenario_id: int,
    pattern: str,
    seed: int,
    ue_count: int,
    duration: int,
    ttt_ms: int,
    hys_db: float,
) -> Path:
    _ensure_local_dirs()

    run_prefix = f"s{scenario_id}_p{pattern}_seed{seed}"
    local_prefix = LOCAL_OUT_DIR / run_prefix

    out_prefix_wsl = _windows_path_to_wsl(local_prefix)
    out_dir_wsl = _windows_path_to_wsl(LOCAL_OUT_DIR)

    ns3_args = (
        "scratch/md_scenarios "
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
            "ns-3 run completed but expected outputs are missing in dataset/:\n"
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
    total = len(scenario_ids) * len(patterns) * len(seeds)
    done = 0

    print(f"Output directory : {LOCAL_OUT_DIR}")
    print(f"WSL distro       : {wsl.distro}")
    print(f"ns-3 root (WSL)  : {wsl.ns3_root}")
    print(f"Matrix           : {len(scenario_ids)} scenarios × {len(patterns)} patterns × {len(seeds)} seeds = {total} runs")
    print(f"UE count         : {ue_count}")
    print(f"Duration         : {duration} (0 = pattern default)")
    print(f"Generator params : tttMs={ttt_ms}, hysDb={hys_db}")

    for scenario_id in scenario_ids:
        for pattern in patterns:
            for seed in seeds:
                done += 1
                run_prefix = f"s{scenario_id}_p{pattern}_seed{seed}"
                print(f"[{done:3d}/{total}] Generating {run_prefix} ...", flush=True)
                run_one(
                    wsl,
                    scenario_id=scenario_id,
                    pattern=pattern,
                    seed=seed,
                    ue_count=ue_count,
                    duration=duration,
                    ttt_ms=ttt_ms,
                    hys_db=hys_db,
                )

    ticks = sorted(LOCAL_OUT_DIR.glob("*_tick.csv"))
    events = sorted(LOCAL_OUT_DIR.glob("*_events.csv"))
    summaries = sorted(LOCAL_OUT_DIR.glob("*_summary.json"))
    print("\nGeneration complete.")
    print(f"  tick files   : {len(ticks)}")
    print(f"  events files : {len(events)}")
    print(f"  summary files: {len(summaries)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase-2 PPO training datasets via WSL ns-3")

    parser.add_argument("--preset", default="full", choices=sorted(PRESETS.keys()))
    parser.add_argument("--scenarioIds", default=None, help="Comma-separated scenario IDs (default from preset)")
    parser.add_argument("--patterns", default=None, help="Comma-separated patterns (A,B,C) (default from preset)")

    parser.add_argument("--seedStart", type=int, default=None)
    parser.add_argument("--seedEnd", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Optional explicit comma-separated seed list")

    parser.add_argument("--ueCount", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--tttMs", type=int, default=None)
    parser.add_argument("--hysDb", type=float, default=None)

    parser.add_argument("--clean", action="store_true", help="Delete existing *_tick/events/summary files from dataset/ first")

    parser.add_argument("--distro", default=os.environ.get("WSL_DISTRO", "Ubuntu"))
    parser.add_argument("--ns3Root", default=os.environ.get("NS3_ROOT", "~/ns-3-dev"))

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    preset = dict(PRESETS[args.preset])

    scenario_ids = _parse_int_list(args.scenarioIds) if args.scenarioIds else _parse_int_list(preset["scenarioIds"])
    patterns = _parse_pattern_list(args.patterns) if args.patterns else _parse_pattern_list(preset["patterns"])

    seed_start = int(args.seedStart) if args.seedStart is not None else int(preset["seedStart"])
    seed_end = int(args.seedEnd) if args.seedEnd is not None else int(preset["seedEnd"])
    explicit_seeds = _parse_int_list(args.seeds) if args.seeds else []
    seeds = _expand_seeds(seed_start, seed_end, explicit_seeds)

    ue_count = int(args.ueCount) if args.ueCount is not None else int(preset["ueCount"])
    duration = int(args.duration) if args.duration is not None else int(preset["duration"])
    ttt_ms = int(args.tttMs) if args.tttMs is not None else int(preset["tttMs"])
    hys_db = float(args.hysDb) if args.hysDb is not None else float(preset["hysDb"])

    wsl = WslConfig(distro=str(args.distro), ns3_root=str(args.ns3Root))

    if args.clean:
        _clean_output_dir()

    deploy_cpp_to_ns3_scratch(wsl)
    generate_matrix(
        wsl,
        scenario_ids=scenario_ids,
        patterns=patterns,
        seeds=seeds,
        ue_count=ue_count,
        duration=duration,
        ttt_ms=ttt_ms,
        hys_db=hys_db,
    )


if __name__ == "__main__":
    main()
