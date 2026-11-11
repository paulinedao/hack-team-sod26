"""Full tracking + Smart Tagging + segment-level possession inference.

Repository location: starter/my_model_segment_full.py
Run from repository root: python starter/my_model_segment_full.py --data-dir /path/to/dataset

Uses no possession ground truth. Requires numpy, pandas and starter/baseline.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from baseline import build, smooth

FPS = 25
ATTACK_TAGS = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
    "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
}


def team_code(name: str) -> int | None:
    value = " ".join(str(name).casefold().strip().split())
    if "barcelona" in value or value in {"bar", "home", "h"}:
        return 0
    if "athletic" in value or "atletic" in value or value in {"ath", "away", "a"}:
        return 1
    return None


def normalize_labels(values: np.ndarray) -> np.ndarray:
    """Accept numeric raw labels or string smoothed labels."""
    arr = np.asarray(values)
    mapping = {0: "H", 1: "A", -1: "D", "0": "H", "1": "A",
               "-1": "D", "H": "H", "A": "A", "D": "D"}
    try:
        return np.array([mapping[x] for x in arr], dtype="<U1")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected baseline labels: {np.unique(arr)}") from exc


def parse_events(xml_path: Path, n_frames: int) -> tuple[list[tuple[int, int, int, str]], pd.DataFrame]:
    root = ET.parse(xml_path).getroot()
    usable = []
    rows = []
    for inst in root.findall(".//instance"):
        code = (inst.findtext("code") or "").strip().upper()
        team_name = ""
        for label in inst.findall(".//label"):
            if (label.findtext("group") or "").strip().casefold() == "team":
                team_name = (label.findtext("text") or "").strip()
                break
        team = team_code(team_name)
        try:
            start_s = float((inst.findtext("start") or "").strip())
            end_s = float((inst.findtext("end") or "").strip())
        except ValueError:
            continue
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue
        start = max(0, int(np.floor(start_s * FPS)))
        end = min(n_frames, int(np.ceil(end_s * FPS)))
        rows.append({"code": code, "team": team_name, "start_s": start_s,
                     "end_s": end_s, "eligible": code in ATTACK_TAGS and team is not None and end > start})
        if code in ATTACK_TAGS and team is not None and end > start:
            usable.append((start, end, team, code))
    return usable, pd.DataFrame(rows)


def tactical_evidence(events: list[tuple[int, int, int, str]], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    diff = np.zeros((n_frames + 1, 2), dtype=np.int32)
    for start, end, team, _ in events:
        diff[start, team] += 1
        diff[end, team] -= 1
    evidence = np.cumsum(diff[:-1], axis=0)
    # Only accept a team when the other has zero simultaneous evidence.
    candidate = np.full(n_frames, -1, dtype=np.int8)
    candidate[(evidence[:, 0] > 0) & (evidence[:, 1] == 0)] = 0
    candidate[(evidence[:, 1] > 0) & (evidence[:, 0] == 0)] = 1
    return candidate, evidence


def local_support(raw: np.ndarray, team: int, radius_frames: int) -> np.ndarray:
    """Share of observed raw labels favoring team in +/- radius window."""
    observed = (raw == 0) | (raw == 1)
    target = raw == team
    c_obs = np.r_[0, np.cumsum(observed.astype(np.int64))]
    c_team = np.r_[0, np.cumsum(target.astype(np.int64))]
    i = np.arange(len(raw))
    lo = np.maximum(0, i - radius_frames)
    hi = np.minimum(len(raw), i + radius_frames + 1)
    denom = c_obs[hi] - c_obs[lo]
    num = c_team[hi] - c_team[lo]
    return np.divide(num, denom, out=np.zeros(len(raw), dtype=float), where=denom > 0)


def apply_smart_tags(raw: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                     min_support: float, radius_seconds: float) -> tuple[np.ndarray, dict]:
    result = baseline.copy()
    radius = round(radius_seconds * FPS)
    h_support = local_support(raw, 0, radius)
    a_support = local_support(raw, 1, radius)
    # Never replace D: tactical phases can overlap dead-ball frames.
    change_h = (candidate == 0) & (baseline == "A") & (h_support >= min_support)
    change_a = (candidate == 1) & (baseline == "H") & (a_support >= min_support)
    result[change_h] = "H"
    result[change_a] = "A"
    return result, {
        "tagged_H_frames": int((candidate == 0).sum()),
        "tagged_A_frames": int((candidate == 1).sum()),
        "conflicting_H_evidence": int(((candidate == 0) & (baseline == "A")).sum()),
        "conflicting_A_evidence": int(((candidate == 1) & (baseline == "H")).sum()),
        "corrected_to_H": int(change_h.sum()),
        "corrected_to_A": int(change_a.sum()),
    }


def get_runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [(int(s), int(e), str(labels[s])) for s, e in zip(starts, ends)]


def refine_segments(labels: np.ndarray, raw: np.ndarray, max_island_seconds: float,
                    min_neighbor_seconds: float, min_track_support: float,
                    max_passes: int) -> tuple[np.ndarray, pd.DataFrame]:
    result = labels.copy()
    changes = []
    max_island = round(max_island_seconds * FPS)
    min_neighbor = round(min_neighbor_seconds * FPS)
    for pass_number in range(1, max_passes + 1):
        runs = get_runs(result)
        proposals = []
        for i in range(1, len(runs) - 1):
            start, end, current = runs[i]
            ls, le, left = runs[i - 1]
            rs, re, right = runs[i + 1]
            if (current not in {"H", "A"} or left != right or left not in {"H", "A"}
                    or current == left or end - start > max_island
                    or le - ls < min_neighbor or re - rs < min_neighbor):
                continue
            observed = raw[start:end]
            observed = observed[(observed == 0) | (observed == 1)]
            if len(observed) == 0:
                continue
            target = 0 if left == "H" else 1
            support = float(np.mean(observed == target))
            if support >= min_track_support:
                proposals.append((start, end, current, left, support))
        for start, end, old, new, support in proposals:
            result[start:end] = new
            changes.append({"pass": pass_number, "start_frame": start,
                            "end_frame": end - 1, "duration_s": (end - start) / FPS,
                            "old": old, "new": new, "tracking_support": support})
        print(f"Segment pass {pass_number}: {len(proposals)} merged")
        if not proposals:
            break
    return result, pd.DataFrame(changes, columns=["pass", "start_frame", "end_frame",
             "duration_s", "old", "new", "tracking_support"])


def describe(labels: np.ndarray) -> dict:
    runs = get_runs(labels)
    durations = np.array([(e - s) / FPS for s, e, label in runs if label in {"H", "A"}])
    return {"possessions": len(durations), "short_under_2s": int((durations < 2).sum()),
            "median_duration_s": round(float(np.median(durations)), 2) if len(durations) else 0,
            "D_percent": round(float(np.mean(labels == "D") * 100), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    default_data = repo.parent / "data/More Than A Hack 2027 - Initial Challenge - Dataset"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--smart-xml", type=Path)
    parser.add_argument("--output", type=Path, default=repo / "evaluation/results/my_model_segment_full.csv")
    parser.add_argument("--diagnostics-dir", type=Path, default=repo / "evaluation/segment_diagnostics")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--tag-min-support", type=float, default=0.35)
    parser.add_argument("--tag-radius-seconds", type=float, default=2.0)
    parser.add_argument("--max-island-seconds", type=float, default=2.0)
    parser.add_argument("--min-neighbor-seconds", type=float, default=3.0)
    parser.add_argument("--min-track-support", type=float, default=0.60)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()

    print("Loading ATD tracking...")
    raw = np.asarray(build(data_dir=str(args.data_dir)))
    if not np.isin(raw, [-1, 0, 1]).all():
        raise ValueError(f"Unexpected raw codes: {np.unique(raw)}")
    baseline = normalize_labels(np.asarray(smooth(raw, args.window)))
    if len(baseline) != len(raw):
        raise ValueError("Smoothed and raw frame counts differ")

    if args.smart_xml:
        xml_path = args.smart_xml
    else:
        xml_files = list(args.data_dir.glob("**/*smart*tag*.xml"))
        if len(xml_files) != 1:
            raise RuntimeError(f"Expected one Smart Tagging XML, found {len(xml_files)}: {xml_files}")
        xml_path = xml_files[0]
    print("Smart Tagging XML:", xml_path)
    events, event_table = parse_events(xml_path, len(raw))
    print(f"Parsed events: {len(event_table)}; eligible: {len(events)}")
    if not events:
        raise RuntimeError("No usable Smart Tagging events found")
    candidate, evidence = tactical_evidence(events, len(raw))
    smart, stats = apply_smart_tags(raw, baseline, candidate,
                                    args.tag_min_support, args.tag_radius_seconds)
    print("Smart Tagging diagnostics:", stats)
    refined, changes = refine_segments(smart, raw, args.max_island_seconds,
                                        args.min_neighbor_seconds, args.min_track_support,
                                        args.max_passes)
    print("Smooth100:", describe(baseline))
    print("Smart Tagging:", describe(smart))
    print("Segment model:", describe(refined))
    print("Smart Tagging changed frames:", int(np.sum(smart != baseline)))
    print("Segment refinement changed frames:", int(np.sum(refined != smart)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(raw)), "label": refined}).to_csv(args.output, index=False)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    changes.to_csv(args.diagnostics_dir / "segment_changes.csv", index=False)
    event_table.to_csv(args.diagnostics_dir / "smart_tagging_events.csv", index=False)
    print("Saved submission:", args.output)
    print("Saved diagnostics:", args.diagnostics_dir)

if __name__ == "__main__":
    main()
