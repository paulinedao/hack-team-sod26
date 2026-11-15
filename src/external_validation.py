"""Validate the possession pipeline on external matches (Sportec / IDSSE open data via kloppy).

The public BAR-ATH ground truth is only 5 minutes, so it cannot be used for tuning. The IDSSE
dataset (7 Bundesliga matches, official DFL tracking at 25 fps) stores, on every frame, whether the
ball is alive and which team owns it -- the same two fields the BAR-ATH ground truth is derived
from. So each external match gives a full-length H/A/D ground truth for free.

This script does NOT modify extract_features.py or predict_possession.py. It reuses their
functions unchanged and only replaces the parts that are specific to BAR-ATH (reading ATD files,
Smart Tagging, games.py entries).

Steps (run them in this order):

    1. check-features  Rebuild BAR-ATH features through features_from_arrays() and compare them,
                       column by column, with extract_features.build(). Proves the array-based
                       path is identical before using it on other data. Needs the raw ATD files.

    2. prepare         Download each Sportec match with kloppy and cache, per frame:
                       player and ball positions in metres (origin at a corner, like ATD), team of
                       each slot, half, and the H/A/D/X ground truth (ball_state + owning team).

    3. profile         Compare the degraded external data with BAR-ATH (ball missing rate and
                       gap lengths, tracked players, ball speed). Adjust the DEGRADATION block
                       until they look alike. DFL data is far cleaner than ATD, so without this
                       the external score is optimistic.

    4. run             Degrade, build features, run predict_possession.predict() unchanged, write
                       predictions + ground truth CSVs, and score with possession_metrics.
"""

import argparse
import ast
import os
import sys
import zlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

# The feature-based pipeline is optional. The direct model-validation flow used by
# starter/my_model_segment_full.py does not depend on extract_features.py or
# predict_possession.py at all, so they must not be imported at module import time.
try:
    import extract_features as ef
except ModuleNotFoundError:
    ef = None

import possession_metrics as pm

try:
    import predict_possession as pp
except ModuleNotFoundError:
    pp = None

from games import DATA_DIR, GAME, GAMES
from starter.baseline import AWAY, HOME, NOBODY

# Sportec / IDSSE matches available through kloppy (sportec.load_open_tracking_data)
SPORTEC_MATCHES = {
    "J03WMX": "1. FC Koeln vs. FC Bayern Muenchen",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Duesseldorf vs. 1. FC Nuernberg",
    "J03WOH": "Fortuna Duesseldorf vs. SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Duesseldorf vs. FC St. Pauli",
    "J03WOY": "Fortuna Duesseldorf vs. F.C. Hansa Rostock",
    "J03WR9": "Fortuna Duesseldorf vs. 1. FC Kaiserslautern",
}

# Paths
CACHE_DIR = os.path.join(DATA_DIR, "external", "sportec")
OUT_DIR = os.path.join(ROOT, "evaluation", "my_results", "external")
REFERENCE_FEATURES = os.path.join(DATA_DIR, "derived", "{}_features_corrected.csv".format(GAME))

# Layout of the external timeline
EXPECTED_FPS = 25             # the feature windows (25/50/10 frames) are counted in frames at 25 fps
HALFTIME_GAP_FRAMES = 250     # untracked, ungraded frames inserted between periods (like BAR-ATH
                              # halftime) so shifts and rolling windows do not run across halves

# Challenge grading windows. The second half is graded from 110,808, but the actual kickoff is at
# 113,742, so those 2,934 frames are still graded as dead-ball and should be treated as D during
# validation. This matches the logic in starter/my_model_segment_full.py and avoids bias when
# comparing external matches with the segment pipeline.
MATCH_GRADE_WINDOWS = (
    (11_185, 90_652),
    (110_808, 190_198),
)
SECOND_HALF_KICKOFF_FRAME = 113_742

# DEGRADATION: make clean DFL tracking look like automatic ATD tracking. Calibrate with profile.
BALL_GAPS_FROM_REFERENCE = True   # ball missing/visible run lengths resampled from BAR-ATH
BALL_NOISE_M = 0.1            # std of slowly varying position error on the ball
PLAYER_NOISE_M = 0.2          # same for players
NOISE_RHO = 0.98              # frame-to-frame correlation of that error (white noise would add
                              # ~sigma * 25 * 1.4 m/s of fake ball speed and break the stationary rule)
PLAYER_TRACKED_FROM_REFERENCE = True  # match BAR-ATH's mean number of tracked players per team
PLAYER_GAP_MEAN_S = 3.0       # mean length of a player dropout
TEAM_ERROR_RATE = 0.0         # share of tracklets assigned to the wrong team (ATD has no IDs, so a
                              # fragment can land in the other team); try e.g. 0.02

# predict_possession settings forced for external data (no Smart Tagging, no BAR-ATH anchors)
FORCED_PREDICT_SETTINGS = {
    "USE_SMART_TAGGING": False,
    "D_BEFORE_KICKOFF": False,
}


# =============================================================================
# Step 1: features from plain arrays
# =============================================================================

def features_from_arrays(players, ball, team_of_track, half, fps, pitch):
    """Same feature table as extract_features.build(), from arrays instead of ATD files.

    players: (n_frames, n_slots, 2) metres, origin at a pitch corner, NaN = not tracked
    ball:    (n_frames, 2) metres, NaN = missing
    team_of_track: (n_slots,) HOME / AWAY
    half:    (n_frames,) 0 = not graded, 1 / 2 = half number

    Every computation is delegated to the functions in extract_features.py, in the order build()
    calls them. Smart Tagging columns are filled with neutral values (no tag anywhere).
    """
    n_frames = len(ball)
    length, width = pitch
    table = pd.DataFrame({"frame": np.arange(n_frames)})
    table["half"] = np.asarray(half, dtype=np.int8)
    table["graded"] = table["half"] > 0
    table["ball_x"] = ball[:, 0]
    table["ball_y"] = ball[:, 1]
    table["ball_missing"] = ~np.isfinite(ball[:, 0]) | ~np.isfinite(ball[:, 1])
    margin = ef.OUT_MARGIN_M
    table["ball_out"] = (
        (ball[:, 0] < -margin)
        | (ball[:, 0] > length + margin)
        | (ball[:, 1] < -margin)
        | (ball[:, 1] > width + margin)
    )
    (
        table["ball_speed"],
        table["ball_speed_smooth"],
        table["ball_accel"],
    ) = ef.ball_motion(ball, fps)
    for name, values in ef.nearest_by_team(players, ball, team_of_track).items():
        table[name] = values
    table["nearest_team"] = ef.nearest_team(
        table["d_home"].to_numpy(),
        table["d_away"].to_numpy(),
    )
    table["margin"] = table["d_away"] - table["d_home"]
    (
        table["frames_since_ball_seen"],
        table["last_ball_x"],
        table["last_ball_y"],
    ) = ef.last_seen(ball)
    ef.add_temporal_features(table)
    table["tag_dead"] = False
    table["tag_set_piece_type"] = ""
    table["tag_goal"] = False
    table["tag_team_poss"] = np.int8(NOBODY)
    table["tag_team_def"] = np.int8(NOBODY)
    return table


def compare_tables(reference, candidate, skip_prefix="tag_"):
    """Column-by-column comparison. Returns a list of (column, problem) for mismatches."""
    problems = []
    for column in reference.columns:
        if column.startswith(skip_prefix):
            continue
        if column not in candidate.columns:
            problems.append((column, "missing"))
            continue
        a = reference[column].to_numpy()
        b = candidate[column].to_numpy()
        if len(a) != len(b):
            problems.append((column, "length {} vs {}".format(len(a), len(b))))
        elif a.dtype.kind == "f" or b.dtype.kind == "f":
            close = np.isclose(a.astype(float), b.astype(float), equal_nan=True, atol=1e-5)
            if not close.all():
                problems.append((column, "{} values differ".format(int((~close).sum()))))
        elif not np.array_equal(a, b):
            problems.append((column, "{} values differ".format(int(np.sum(a != b)))))
    return problems


def require_extract_features():
    if ef is None:
        raise RuntimeError(
            "extract_features.py is not present in this workspace, so the feature-based checks are unavailable. "
            "Use the direct submission validation path instead: "
            "python src/external_validation.py validate-submission --match <MATCH_ID> <submission.csv>"
        )
    return ef


def require_predict_possession():
    if pp is None:
        raise RuntimeError(
            "predict_possession.py is not present in this workspace. The direct model-validation path does not need it; "
            "run validate-submission instead."
        )
    return pp


def cmd_check_features(args):
    """Step 1: features_from_arrays() on BAR-ATH must equal extract_features.build()."""
    from games import require, source_frames
    from starter.baseline import team_of_each_track

    ef = require_extract_features()
    game = args.game
    print("building reference with extract_features.build() ...")
    reference = ef.build(game=game)

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    track_teams = team_of_each_track(meta_path)
    pitch = GAMES[game]["pitch"]
    players, ball = ef.read_atd(raw_path, source_frames("tactical", game), len(track_teams), pitch)
    candidate = features_from_arrays(
        players, ball, track_teams, ef.half_mask(game, len(ball)), GAMES[game]["fps"], pitch
    )

    problems = compare_tables(reference, candidate)
    if problems:
        print("\nMISMATCH in {} column(s):".format(len(problems)))
        for column, problem in problems:
            print("  {:<28} {}".format(column, problem))
        sys.exit(1)
    print("\nOK: all {} non-tag columns identical ({} frames).".format(
        sum(not c.startswith("tag_") for c in reference.columns), len(reference)))


# =============================================================================
# Step 2: Sportec loader and ground truth
# =============================================================================

def cache_path(match_id):
    return os.path.join(CACHE_DIR, "{}.npz".format(match_id))


def to_corner_metres(value, dim, size):
    """Shift a kloppy metric coordinate so the pitch spans 0..size (ATD convention)."""
    low = dim.min if dim.min is not None else -size / 2.0
    return value - low


def sportec_arrays(match_id):
    """Load one IDSSE match with kloppy and turn it into the arrays the pipeline needs.

    Returns a dict with players, ball, team_of_track, half, labels (H/A/D/X), fps, pitch, names.
    Ground truth per frame: D if ball_state is dead; else H/A from ball_owning_team; X if unknown.
    """
    from kloppy import sportec
    from kloppy.domain import BallState, Ground

    print("loading {} ({}) with kloppy, this downloads and parses the full match ...".format(
        match_id, SPORTEC_MATCHES.get(match_id, "?")))
    dataset = sportec.load_open_tracking_data(match_id=match_id, only_alive=False, coordinates="sportec")
    meta = dataset.metadata

    fps = int(round(meta.frame_rate))
    if fps != EXPECTED_FPS:
        raise ValueError("{}: {} fps, the feature windows assume {}".format(
            match_id, fps, EXPECTED_FPS))
    dims = meta.pitch_dimensions
    length = float(dims.pitch_length)
    width = float(dims.pitch_width)
    if dims.x_dim.max is None or abs(dims.x_dim.max - dims.x_dim.min) < 10:
        raise ValueError("{}: pitch x range {}..{} is not in metres".format(match_id, dims.x_dim.min, dims.x_dim.max))

    slots = []
    team_of_track = []
    for team in meta.teams:
        code = HOME if team.ground == Ground.HOME else AWAY
        for player in team.players:
            slots.append(player)
            team_of_track.append(code)
    slot_of = {player: i for i, player in enumerate(slots)}

    # Frames grouped by period, with an ungraded gap between periods.
    periods = sorted({frame.period.id for frame in dataset.frames})
    n_frames = sum(1 for _ in dataset.frames) + HALFTIME_GAP_FRAMES * (len(periods) - 1)
    players = np.full((n_frames, len(slots), 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    half = np.zeros(n_frames, dtype=np.int8)
    labels = np.full(n_frames, "X", dtype="<U1")

    row = 0
    previous_period = None
    for frame in dataset.frames:
        period = frame.period.id
        if previous_period is not None and period != previous_period:
            row += HALFTIME_GAP_FRAMES
        previous_period = period
        half[row] = period

        for player, data in frame.players_data.items():
            point = data.coordinates
            if point is None or player not in slot_of:
                continue
            players[row, slot_of[player]] = (
                to_corner_metres(point.x, dims.x_dim, length),
                to_corner_metres(point.y, dims.y_dim, width),
            )
        point = frame.ball_coordinates
        if point is not None:
            ball[row] = (
                to_corner_metres(point.x, dims.x_dim, length),
                to_corner_metres(point.y, dims.y_dim, width),
            )

        if frame.ball_state == BallState.DEAD:
            labels[row] = "D"
        elif frame.ball_state == BallState.ALIVE and frame.ball_owning_team is not None:
            labels[row] = "H" if frame.ball_owning_team.ground == Ground.HOME else "A"
        row += 1

    names = [team.name for team in meta.teams]
    return {
        "players": players[:row],
        "ball": ball[:row],
        "team_of_track": np.array(team_of_track, dtype=np.int8),
        "half": half[:row],
        "labels": labels[:row],
        "fps": np.int16(fps),
        "pitch": np.array([length, width]),
        "names": np.array(names),
    }


def load_match(match_id):
    """Cached arrays for one match (run prepare first)."""
    path = cache_path(match_id)
    if not os.path.isfile(path):
        sys.exit("{} not cached; run: python src/external_validation.py prepare --matches {}"
                 .format(match_id, match_id))
    with np.load(path) as data:
        match = {key: data[key] for key in data.files}
    match["fps"] = int(match["fps"])
    match["pitch"] = tuple(float(v) for v in match["pitch"])
    return match


def gt_summary(labels):
    """What the ground truth looks like: D share, possession count and mean length."""
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    graded = codes != pm.X
    stats = pm.possession_stats(codes, runs)
    minutes = graded.sum() / pm.FPS / 60.0
    return {
        "graded_min": minutes,
        "d_share": float(np.mean(codes[graded] == pm.D)) if graded.any() else np.nan,
        "n_spells": stats["n_spells"],
        "spells_per_min": stats["n_spells"] / minutes if minutes else np.nan,
        "mean_spell_s": stats["mean_duration_s"],
        "home_share": stats["home_share"],
    }


def print_gt_summary(name, summary):
    print("  {:<24} {:6.1f} min  D {:5.1%}  {:4d} possessions  {:4.2f}/min  mean {:5.1f} s".format(
        name, summary["graded_min"], summary["d_share"], summary["n_spells"],
        summary["spells_per_min"], summary["mean_spell_s"]))


def cmd_prepare(args):
    """Step 2: download, convert and cache each match; print its ground-truth profile."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for match_id in args.matches:
        path = cache_path(match_id)
        if os.path.isfile(path) and not args.force:
            print("{} already cached ({})".format(match_id, path))
            match = load_match(match_id)
        else:
            match = sportec_arrays(match_id)
            np.savez_compressed(path, **match)
            print("wrote {}".format(path))
        print("  teams: H = {}, A = {}; {} frames, {} slots, pitch {}".format(
            match["names"][0], match["names"][1], len(match["ball"]),
            len(match["team_of_track"]), match["pitch"]))
        print_gt_summary(match_id, gt_summary(match["labels"]))
    if args.reference_gt:
        print("\nreference (compare the definitions):")
        print_gt_summary(GAME + " 5-min GT", gt_summary(pm.read_label_csv(args.reference_gt)["label"]))


# =============================================================================
# Step 3: degradation, calibrated on BAR-ATH
# =============================================================================

def run_lengths(mask):
    """Lengths of the True runs and of the False runs of a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return np.array([], int), np.array([], int)
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(mask)]))
    lengths = np.diff(bounds)
    first = mask[0]
    true_runs = lengths[0::2] if first else lengths[1::2]
    false_runs = lengths[1::2] if first else lengths[0::2]
    return true_runs, false_runs


def reference_profile(features_path):
    """Statistics of the real (automatic) BAR-ATH tracking on graded frames."""
    table = pd.read_csv(features_path, dtype={"tag_set_piece_type": "object"})
    missing_runs, visible_runs = [], []
    for _, block in table[pp.as_bool(table["graded"])].groupby("half"):
        missing, visible = run_lengths(pp.as_bool(block["ball_missing"]))
        missing_runs.append(missing)
        visible_runs.append(visible)
    graded = table[pp.as_bool(table["graded"])]
    return {
        "missing_runs": np.concatenate(missing_runs),
        "visible_runs": np.concatenate(visible_runs),
        "n_home_tracked": float(graded["n_home_tracked"].mean()),
        "n_away_tracked": float(graded["n_away_tracked"].mean()),
        "table": graded,
    }


def alternating_mask(n, on_sampler, off_sampler, rng, start_on=True):
    """Boolean mask built from alternating on/off runs drawn from the two samplers."""
    mask = np.zeros(n, dtype=bool)
    position = 0
    on = start_on
    while position < n:
        length = max(1, int(on_sampler(rng) if on else off_sampler(rng)))
        if on:
            mask[position:position + length] = True
        position += length
        on = not on
    return mask


def correlated_noise(shape, sigma, rho, rng):
    """AR(1) noise along axis 0 with stationary std sigma: x_t = rho x_(t-1) + e_t."""
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    # The ewm below scales the innovation by (1 - rho); this std gives the target sigma after it.
    innovation = sigma * np.sqrt((1 + rho) / (1 - rho))
    flat = rng.normal(0.0, innovation, size=(shape[0], int(np.prod(shape[1:]))))
    smooth = pd.DataFrame(flat).ewm(alpha=1 - rho, adjust=False).mean().to_numpy()
    return smooth.reshape(shape).astype(np.float32)


def degrade(match, reference, rng):
    """Copy of the match arrays with ATD-like noise. Labels are untouched."""
    players = match["players"].copy()
    ball = match["ball"].copy()
    team_of_track = match["team_of_track"]
    half = match["half"]
    n, n_slots, _ = players.shape
    fps = match["fps"]

    # Position error, slowly varying (NaN stays NaN).
    ball += correlated_noise(ball.shape, BALL_NOISE_M, NOISE_RHO, rng)
    players += correlated_noise(players.shape, PLAYER_NOISE_M, NOISE_RHO, rng)

    # Ball gaps: alternate visible / missing runs with BAR-ATH's run lengths.
    if BALL_GAPS_FROM_REFERENCE:
        visible = alternating_mask(
            n,
            lambda r: r.choice(reference["visible_runs"]),
            lambda r: r.choice(reference["missing_runs"]),
            rng,
            start_on=bool(rng.random() < 0.5),
        )
        ball[~visible] = np.nan

    # Player dropouts, so the mean number of tracked players per team matches BAR-ATH.
    ghost = np.full_like(players, np.nan)
    if PLAYER_TRACKED_FROM_REFERENCE or TEAM_ERROR_RATE > 0:
        graded = half > 0
        tracked = np.isfinite(players[graded, :, 0])
        off_mean = PLAYER_GAP_MEAN_S * fps
        for code, key in ((HOME, "n_home_tracked"), (AWAY, "n_away_tracked")):
            team_slots = np.flatnonzero(team_of_track == code)
            clean_mean = tracked[:, team_slots].sum(axis=1).mean()
            keep = 1.0
            if PLAYER_TRACKED_FROM_REFERENCE and clean_mean > 0:
                keep = min(1.0, reference[key] / clean_mean)
            on_mean = off_mean * keep / (1 - keep) if keep < 1 else np.inf
            for slot in team_slots:
                if np.isfinite(on_mean):
                    on = alternating_mask(
                        n,
                        lambda r: r.geometric(1.0 / on_mean),
                        lambda r: r.geometric(1.0 / off_mean),
                        rng,
                    )
                    players[~on, slot] = np.nan
                else:
                    on = np.ones(n, dtype=bool)
                if TEAM_ERROR_RATE > 0:
                    # A tracklet = one on-run; move a share of them to the other team's ghost slot.
                    starts = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8)))) == 1)
                    stops = np.flatnonzero(np.diff(np.concatenate((on.astype(np.int8), [0]))) == -1) + 1
                    for a, b in zip(starts, stops):
                        if rng.random() < TEAM_ERROR_RATE:
                            ghost[a:b, slot] = players[a:b, slot]
                            players[a:b, slot] = np.nan

    if TEAM_ERROR_RATE > 0:
        players = np.concatenate([players, ghost], axis=1)
        flipped = np.where(team_of_track == HOME, AWAY, HOME).astype(np.int8)
        team_of_track = np.concatenate([team_of_track, flipped])

    out = dict(match)
    out.update(players=players, ball=ball, team_of_track=team_of_track)
    return out


def match_rng(match_id, seed):
    return np.random.default_rng([seed, zlib.crc32(match_id.encode())])


def features_for(match, reference, match_id, seed, clean):
    ef = require_extract_features()
    data = match if clean else degrade(match, reference, match_rng(match_id, seed))
    return features_from_arrays(
        data["players"], data["ball"], data["team_of_track"], data["half"],
        data["fps"], data["pitch"],
    )


def tracking_profile(table):
    """Tracking-quality statistics of a feature table, graded frames only."""
    graded = table[pp.as_bool(table["graded"])]
    missing = pp.as_bool(graded["ball_missing"])
    missing_runs, _ = run_lengths(missing)
    speed = graded.loc[~missing, "ball_speed_smooth"].dropna()
    return {
        "ball missing": "{:.1%}".format(missing.mean()),
        "ball gap median (s)": "{:.2f}".format(np.median(missing_runs) / 25 if len(missing_runs) else 0),
        "ball gap mean (s)": "{:.2f}".format(np.mean(missing_runs) / 25 if len(missing_runs) else 0),
        "ball outside pitch": "{:.1%}".format(pp.as_bool(graded["ball_out"]).mean()),
        "home tracked (mean)": "{:.1f}".format(graded["n_home_tracked"].mean()),
        "away tracked (mean)": "{:.1f}".format(graded["n_away_tracked"].mean()),
        "ball speed p10 (m/s)": "{:.2f}".format(speed.quantile(0.10)),
        "ball speed median": "{:.2f}".format(speed.median()),
        "ball speed p90": "{:.2f}".format(speed.quantile(0.90)),
        "nearest player median (m)": "{:.1f}".format(
            np.fmin(graded["d_home"], graded["d_away"]).median()),
    }


def cmd_profile(args):
    """Step 3: side-by-side tracking statistics, BAR-ATH vs external (clean and degraded)."""
    require_extract_features()
    reference = reference_profile(args.reference_features)
    columns = {GAME: tracking_profile(reference["table"])}
    for match_id in args.matches:
        match = load_match(match_id)
        columns[match_id + " clean"] = tracking_profile(
            features_for(match, reference, match_id, args.seed, clean=True))
        columns[match_id + " degraded"] = tracking_profile(
            features_for(match, reference, match_id, args.seed, clean=False))
    print(pd.DataFrame(columns).to_string())
    print("\nTune the DEGRADATION block until the degraded columns resemble {}.".format(GAME))


# =============================================================================
# Step 4: predict and score
# =============================================================================

def apply_predict_settings(overrides):
    """Force external-data settings, then apply --set NAME=VALUE overrides to predict_possession."""
    pp = require_predict_possession()
    settings = dict(FORCED_PREDICT_SETTINGS)
    for item in overrides:
        name, _, value = item.partition("=")
        if not hasattr(pp, name):
            sys.exit("predict_possession has no parameter {!r}".format(name))
        settings[name] = ast.literal_eval(value)
    for name, value in settings.items():
        setattr(pp, name, value)
    return settings


def register_game(key, match):
    """predict() reads fps and pitch from games.GAMES; add an entry for this match (in memory)."""
    GAMES[key] = {"fps": match["fps"], "pitch": match["pitch"]}


def apply_challenge_grading(labels, second_half_kickoff_frame=SECOND_HALF_KICKOFF_FRAME):
    """Coerce labels to the challenge's graded windows and delayed second-half kickoff.

    Frames outside the two graded halves are X. Within the second half, frames before the actual
    kickoff are still part of the graded window and must be scored as D, which matches the
    segment-model fix in starter/my_model_segment_full.py.
    """
    labels = np.asarray(labels)
    out = np.full(len(labels), pm.X, dtype="<U1")
    for start, end in MATCH_GRADE_WINDOWS:
        out[start:end] = labels[start:end]
    out[110_808:second_half_kickoff_frame] = "D"
    return out


def score(labels, predictions, bootstrap_n, use_delayed_kickoff=True):
    if use_delayed_kickoff:
        labels = apply_challenge_grading(labels)
        predictions = apply_challenge_grading(predictions)
    gt = pm.label_codes(labels)
    pred = pm.label_codes(predictions)
    result = pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)
    row = {
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "F1_H": result["per_class"]["H"]["f1"],
        "F1_A": result["per_class"]["A"]["f1"],
        "F1_D": result["per_class"]["D"]["f1"],
        "D_prec": result["per_class"]["D"]["precision"],
        "D_rec": result["per_class"]["D"]["recall"],
        "team_acc": result["team_acc_both_alive"],
        "gt_spells": result["n_gt_spells"],
        "pred_spells": result["n_pred_spells"],
        "gt_mean_s": result["gt_mean_spell_duration_s"],
        "pred_mean_s": result["mean_spell_duration_s"],
    }
    if bootstrap_n:
        row["spellF1_low"] = result["spell_bootstrap_low"]
        row["spellF1_high"] = result["spell_bootstrap_high"]
    return row


def cmd_validate_submission(args):
    """Validate a CSV submission from a model such as starter/my_model_segment_full.py."""
    match = load_match(args.match)
    table = pd.read_csv(args.submission).sort_values("frame").reset_index(drop=True)
    if not {"frame", "label"}.issubset(table.columns):
        raise ValueError("submission CSV must contain 'frame' and 'label' columns")
    submission = table["label"].to_numpy()
    if len(submission) != len(match["labels"]):
        raise ValueError(
            "submission length {} does not match match length {} for {}".format(
                len(submission), len(match["labels"]), args.match))
    use_delay = not args.no_delay_kickoff
    row = {"match": args.match, "seed": "submission"}
    row.update(score(match["labels"], submission, args.bootstrap, use_delayed_kickoff=use_delay))
    print("{}: spellF1 {:.3f}  macroF1 {:.3f}".format(args.match, row["spellF1"], row["macroF1"]))
    print(pd.DataFrame([row]).to_string(index=False, float_format="{:.3f}".format))


def cmd_run(args):
    """Step 4: degrade -> features -> predict_possession.predict() -> possession_metrics."""
    pp = require_predict_possession()
    settings = apply_predict_settings(args.set)
    print("predict_possession overrides: {}".format(settings))
    reference = None if args.clean else reference_profile(args.reference_features)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.tag or ("clean" if args.clean else "degraded")

    rows = []
    for match_id in args.matches:
        match = load_match(match_id)
        key = "sportec-" + match_id
        register_game(key, match)
        pd.DataFrame({"frame": np.arange(len(match["labels"])), "label": match["labels"]}).to_csv(
            os.path.join(OUT_DIR, "{}_gt.csv".format(key)), index=False)

        for seed in ([0] if args.clean else args.seeds):
            table = features_for(match, reference, match_id, seed, args.clean)
            out = pp.predict(table, key)
            reason = pp.validate(out, len(table))
            if reason:
                sys.exit("{}: invalid prediction: {}".format(key, reason))
            if args.save_predictions:
                out[pp.SUBMISSION_COLUMNS].to_csv(os.path.join(
                    OUT_DIR, "{}_{}_seed{}_predictions.csv".format(key, tag, seed)), index=False)
            row = {"match": match_id, "seed": seed}
            row.update(score(match["labels"], out["label"].to_numpy(), args.bootstrap,
                            use_delayed_kickoff=not args.no_delay_kickoff))
            rows.append(row)
            print("{} seed {}: spellF1 {:.3f}  macroF1 {:.3f}".format(
                match_id, seed, row["spellF1"], row["macroF1"]))

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== PER MATCH / SEED ===")
    print(results.to_string(index=False, float_format="{:.3f}".format))
    print("\n=== MEAN OVER MATCHES (std across matches) ===")
    per_match = results.drop(columns="seed").groupby("match").mean()
    summary = pd.DataFrame({"mean": per_match.mean(), "std": per_match.std()})
    print(summary.to_string(float_format="{:.3f}".format))

    path = os.path.join(OUT_DIR, "summary_{}.csv".format(tag))
    results.to_csv(path, index=False)
    print("\nWrote {}".format(path))


# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-features", help="step 1: verify features_from_arrays")
    check.add_argument("--game", default=GAME)
    check.set_defaults(func=cmd_check_features)

    prepare = commands.add_parser("prepare", help="step 2: download and cache Sportec matches")
    prepare.add_argument("--matches", nargs="+", default=list(SPORTEC_MATCHES))
    prepare.add_argument("--force", action="store_true", help="re-download even if cached")
    prepare.add_argument("--reference-gt", help="BAR-ATH 5-min GT csv, to compare definitions")
    prepare.set_defaults(func=cmd_prepare)

    profile = commands.add_parser("profile", help="step 3: compare tracking quality with BAR-ATH")
    profile.add_argument("--matches", nargs="+", default=["J03WMX"])
    profile.add_argument("--reference-features", default=REFERENCE_FEATURES)
    profile.add_argument("--seed", type=int, default=0)
    profile.set_defaults(func=cmd_profile)

    run = commands.add_parser("run", help="step 4: predict and score external matches")
    run.add_argument("--matches", nargs="+", default=list(SPORTEC_MATCHES))
    run.add_argument("--seeds", nargs="+", type=int, default=[0],
                     help="degradation seeds; several give the spread due to noise alone")
    run.add_argument("--clean", action="store_true", help="no degradation (upper bound)")
    run.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                     help="override a predict_possession parameter, e.g. --set MIN_SPELL_S=3.0")
    run.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws for spellF1 interval")
    run.add_argument("--reference-features", default=REFERENCE_FEATURES)
    run.add_argument("--save-predictions", action="store_true")
    run.add_argument("--tag", help="name of this run in the output files")
    run.add_argument("--no-delay-kickoff", action="store_true",
                     help="disable the delayed-kickoff D correction before scoring")
    run.set_defaults(func=cmd_run)

    validate = commands.add_parser("validate-submission",
                                   help="score a submission CSV such as the one from my_model_segment_full.py")
    validate.add_argument("--match", required=True, choices=list(SPORTEC_MATCHES),
                          help="external match ID to compare against")
    validate.add_argument("submission", help="CSV with frame,label columns")
    validate.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws for spellF1 interval")
    validate.add_argument("--no-delay-kickoff", action="store_true",
                          help="disable the delayed-kickoff D correction before scoring")
    validate.set_defaults(func=cmd_validate_submission)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()