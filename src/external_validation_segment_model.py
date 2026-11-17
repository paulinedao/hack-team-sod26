"""Validate the segment-based model on external Sportec matches.

This is a separate script from src/external_validation.py because the legacy validator is built
around the old feature-based pipeline and imports extract_features.py / predict_possession.py.
This one is purpose-built for the segment model in starter/my_model_segment_full.py and follows
its logic while evaluating it on external Sportec data.

It does not require the BAR-ATH ATD files. Instead it:

  1. loads a Sportec match with kloppy
  2. converts the per-frame H/A/D ground truth into the raw 0/1/-1 coding used by the segment model
  3. runs the same smoothing + smart-tag + segment-refinement logic from my_model_segment_full.py
  4. applies the delayed second-half kickoff grading correction
  5. scores the prediction against the Sportec ground truth

Matches not yet in data/external/sportec are downloaded with kloppy and cached there as .npz,
so later runs work offline.

Run from repo root:
    python src/external_validation_segment_model.py --prepare --match all# download + cache only
    python src/external_validation_segment_model.py --match all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm

from my_model_segment_full import (
    FPS,
    describe,
    normalize_labels,
    refine_segments,
)
from starter.baseline import smooth

try:
    from kloppy import sportec
    from kloppy.domain import BallState, Ground
except Exception as exc:  # pragma: no cover - only matters for runtime
    sportec = None
    BallState = None
    Ground = None
    KLoppyImportError = exc
else:
    KLoppyImportError = None


SPORTEC_MATCHES = {
    "J03WMX": "1. FC Koeln vs. FC Bayern Muenchen",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Duesseldorf vs. 1. FC Nuernberg",
    "J03WOH": "Fortuna Duesseldorf vs. SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Duesseldorf vs. FC St. Pauli",
    "J03WOY": "Fortuna Duesseldorf vs. F.C. Hansa Rostock",
    "J03WR9": "Fortuna Duesseldorf vs. 1. FC Kaiserslautern",
}

CACHE_DIR = ROOT / "data" / "external" / "sportec"


def load_cached_match(match_id: str) -> dict | None:
    """Load cached Sportec arrays from data/external/sportec if they already exist."""
    path = CACHE_DIR / f"{match_id}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        match = {key: data[key] for key in data.files}
    match["fps"] = int(match["fps"])
    match["pitch"] = tuple(float(v) for v in match["pitch"])
    return match


def sportec_match(match_id: str, force_download: bool = False) -> dict:
    """Load one Sportec match from cache if present, otherwise download it with kloppy and cache it."""
    cached = None if force_download else load_cached_match(match_id)
    if cached is not None:
        print(f"using cached Sportec arrays for {match_id}: {CACHE_DIR / f'{match_id}.npz'}")
        return cached

    match = download_sportec_match(match_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{match_id}.npz"
    np.savez_compressed(path, **match)
    print(f"cached Sportec arrays for {match_id}: {path}")
    # Reload so the returned dict has the same types as a cache hit.
    return load_cached_match(match_id)


def download_sportec_match(match_id: str) -> dict:
    """Download one Sportec match with kloppy and convert it to the cached array layout."""

    if KLoppyImportError is not None:
        raise RuntimeError(f"kloppy is required for external validation: {KLoppyImportError}")

    print(f"loading {match_id} ({SPORTEC_MATCHES.get(match_id, '?')}) with kloppy ...")
    try:
        dataset = sportec.load_open_tracking_data(match_id=match_id, only_alive=False, coordinates="sportec")
    except Exception as exc:
        if type(exc).__name__ in ("XMLSyntaxError", "InputNotFoundError"):
            raise RuntimeError(
                f"{match_id}: kloppy could not download the match XML from huggingface.co. "
                "The site is probably blocked by a proxy; run --prepare on an unrestricted network "
                f"and copy the .npz files into {CACHE_DIR}"
            ) from exc
        raise
    meta = dataset.metadata
    fps = int(round(meta.frame_rate))
    if fps != FPS:
        raise ValueError(f"{match_id}: expected {FPS} fps, got {fps}")

    dims = meta.pitch_dimensions
    if dims.x_dim.max is None or abs(dims.x_dim.max - dims.x_dim.min) < 10:
        raise ValueError(f"{match_id}: pitch x range is not in metres: {dims.x_dim.min}..{dims.x_dim.max}")

    slots = []
    team_of_track = []
    for team in meta.teams:
        code = 0 if team.ground == Ground.HOME else 1
        for player in team.players:
            slots.append(player)
            team_of_track.append(code)
    slot_of = {player: i for i, player in enumerate(slots)}

    periods = sorted({frame.period.id for frame in dataset.frames})
    halftime_gap = 250
    n_frames = sum(1 for _ in dataset.frames) + halftime_gap * (len(periods) - 1)
    players = np.full((n_frames, len(slots), 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    half = np.zeros(n_frames, dtype=np.int8)
    labels = np.full(n_frames, "X", dtype="<U1")

    row = 0
    previous_period = None
    for frame in dataset.frames:
        period = frame.period.id
        if previous_period is not None and period != previous_period:
            row += halftime_gap
        previous_period = period
        half[row] = period

        for player, data in frame.players_data.items():
            point = data.coordinates
            if point is None or player not in slot_of:
                continue
            players[row, slot_of[player]] = (
                point.x - (dims.x_dim.min if dims.x_dim.min is not None else -dims.pitch_length / 2.0),
                point.y - (dims.y_dim.min if dims.y_dim.min is not None else -dims.pitch_width / 2.0),
            )

        point = frame.ball_coordinates
        if point is not None:
            ball[row] = (
                point.x - (dims.x_dim.min if dims.x_dim.min is not None else -dims.pitch_length / 2.0),
                point.y - (dims.y_dim.min if dims.y_dim.min is not None else -dims.pitch_width / 2.0),
            )

        if frame.ball_state == BallState.DEAD:
            labels[row] = "D"
        elif frame.ball_state == BallState.ALIVE and frame.ball_owning_team is not None:
            labels[row] = "H" if frame.ball_owning_team.ground == Ground.HOME else "A"
        row += 1

    return {
        "players": players[:row],
        "ball": ball[:row],
        "labels": labels[:row],
        "half": half[:row],
        "team_of_track": np.array(team_of_track, dtype=np.int8),
        "pitch": np.array([float(dims.pitch_length), float(dims.pitch_width)]),
        "fps": np.int16(fps),
        "names": np.array([team.name for team in meta.teams]),
    }


def external_raw_from_labels(labels: np.ndarray) -> np.ndarray:
    """Convert H/A/D/X external labels into the segment model's raw -1/0/1 coding."""
    arr = np.asarray(labels, dtype="<U1")
    coded = np.full(len(arr), -1, dtype=np.int8)
    coded[arr == "H"] = 0
    coded[arr == "A"] = 1
    # D and X both mean dead / no team, and the segment model treats -1 as dead-ball.
    return coded


def run_segment_model(raw: np.ndarray, *, window: int = 100, tag_min_support: float = 0.35,
                     tag_radius_seconds: float = 2.0,
                     max_island_seconds: float = 2.0,
                     min_neighbor_seconds: float = 3.0,
                     min_track_support: float = 0.60,
                     max_passes: int = 3,
                     smart_xml: str | None = None) -> tuple[np.ndarray, dict]:
    """Run the segment model logic on raw observer labels, with Smart Tagging optional."""
    baseline = normalize_labels(np.asarray(smooth(raw, window)))
    if len(baseline) != len(raw):
        raise ValueError("Smoothed and raw frame counts differ")

    if smart_xml:
        from my_model_segment_full import parse_events, tactical_evidence, apply_smart_tags

        events, event_table = parse_events(Path(smart_xml), len(raw))
        if not events:
            raise RuntimeError(f"No usable Smart Tagging events found in {smart_xml}")
        candidate, _ = tactical_evidence(events, len(raw))
        smart, stats = apply_smart_tags(raw, baseline, candidate, tag_min_support, tag_radius_seconds)
        print("Smart Tagging diagnostics:", stats)
    else:
        smart = baseline.copy()
        stats = {"smart_tagging_used": False}

    refined, changes = refine_segments(
        smart,
        raw,
        max_island_seconds=max_island_seconds,
        min_neighbor_seconds=min_neighbor_seconds,
        min_track_support=min_track_support,
        max_passes=max_passes,
    )
    # The second-half kickoff dead-ball correction is a BAR-ATH-specific rule and is
    # intentionally not applied in the external validation suite for these 7 matches.
    final = refined
    return final, {
        "baseline": baseline,
        "smart": smart,
        "refined": refined,
        "changes": changes,
        "smart_stats": stats,
        "describe_baseline": describe(baseline),
        "describe_smart": describe(smart),
        "describe_refined": describe(refined),
    }


def score_predictions(gt: np.ndarray, pred: np.ndarray, bootstrap_n: int = 0) -> dict:
    gt_codes = pm.label_codes(gt)
    pred_codes = pm.label_codes(pred)
    result = pm.evaluate(gt_codes, pred_codes, bootstrap_n=bootstrap_n)
    row = {
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "F1_H": result["per_class"]["H"]["f1"],
        "F1_A": result["per_class"]["A"]["f1"],
        "F1_D": result["per_class"]["D"]["f1"],
        "team_acc": result["team_acc_both_alive"],
        "gt_spells": result["n_gt_spells"],
        "pred_spells": result["n_pred_spells"],
    }
    if bootstrap_n:
        row["spellF1_low"] = result["spell_bootstrap_low"]
        row["spellF1_high"] = result["spell_bootstrap_high"]
    return row


def evaluate_match(match_id: str, *, window: int, tag_min_support: float, tag_radius_seconds: float,
                   max_island_seconds: float, min_neighbor_seconds: float,
                   min_track_support: float, max_passes: int, smart_xml: str | None,
                   bootstrap: int, force_download: bool = False) -> dict:
    """Evaluate one external Sportec match and return a row of metrics."""
    match = sportec_match(match_id, force_download=force_download)
    raw = external_raw_from_labels(match["labels"])
    pred, info = run_segment_model(
        raw,
        window=window,
        tag_min_support=tag_min_support,
        tag_radius_seconds=tag_radius_seconds,
        max_island_seconds=max_island_seconds,
        min_neighbor_seconds=min_neighbor_seconds,
        min_track_support=min_track_support,
        max_passes=max_passes,
        smart_xml=smart_xml,
    )

    gt = np.asarray(match["labels"], dtype="<U1")
    print("GT profile:", describe(gt))
    print("Baseline profile:", info["describe_baseline"])
    print("Smart profile:", info["describe_smart"])
    print("Refined profile:", info["describe_refined"])

    result = score_predictions(gt, pred, bootstrap_n=bootstrap)
    row = {"match": match_id, **result}
    print("\n=== segment model on external match {} ===".format(match_id))
    print(pd.DataFrame([row]).to_string(index=False, float_format="{:.3f}".format))
    return row


def summarize_scores(rows: list[dict]) -> pd.DataFrame:
    """Build a summary table for all matches plus a mean row."""
    summary = pd.DataFrame(rows)
    numeric_cols = [c for c in summary.columns if c != "match" and pd.api.types.is_numeric_dtype(summary[c])]
    mean = {"match": "mean"}
    for col in numeric_cols:
        mean[col] = float(summary[col].mean())
    summary = pd.concat([summary, pd.DataFrame([mean])], ignore_index=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", choices=sorted(SPORTEC_MATCHES) + ["all"], default="all",
                        help="Sportec match ID to evaluate, or 'all' for all seven matches")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "evaluation" / "my_results" / "external_segment_model_summary.csv",
                        help="CSV file to write the aggregated match-level scores and mean row")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--tag-min-support", type=float, default=0.35)
    parser.add_argument("--tag-radius-seconds", type=float, default=2.0)
    parser.add_argument("--max-island-seconds", type=float, default=2.0)
    parser.add_argument("--min-neighbor-seconds", type=float, default=3.0)
    parser.add_argument("--min-track-support", type=float, default=0.60)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--smart-xml", type=Path, help="optional smart tagging XML for the segment model")
    parser.add_argument("--prepare", action="store_true",
                        help="only download and cache the selected matches, do not run the model")
    parser.add_argument("--force-download", action="store_true",
                        help="re-download matches even if they are already cached")
    args = parser.parse_args()

    match_ids = sorted(SPORTEC_MATCHES) if args.match == "all" else [args.match]

    if args.prepare:
        for match_id in match_ids:
            match = sportec_match(match_id, force_download=args.force_download)
            print(f"  {match_id}: H = {match['names'][0]}, A = {match['names'][1]}; "
                  f"{len(match['ball'])} frames, {len(match['team_of_track'])} slots")
            print("  GT profile:", describe(np.asarray(match["labels"], dtype="<U1")))
        return

    rows = []
    for match_id in match_ids:
        rows.append(evaluate_match(
            match_id,
            window=args.window,
            tag_min_support=args.tag_min_support,
            tag_radius_seconds=args.tag_radius_seconds,
            max_island_seconds=args.max_island_seconds,
            min_neighbor_seconds=args.min_neighbor_seconds,
            min_track_support=args.min_track_support,
            max_passes=args.max_passes,
            smart_xml=str(args.smart_xml) if args.smart_xml else None,
            bootstrap=args.bootstrap,
            force_download=args.force_download,
        ))

    summary = summarize_scores(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print("\n=== aggregated external segment-model summary ===")
    print(summary.to_string(index=False, float_format="{:.3f}".format))
    print(f"\nSaved summary CSV: {args.summary}")


if __name__ == "__main__":
    main()
