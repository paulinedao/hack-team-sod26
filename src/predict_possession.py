"""Predict possession labels (H / A / D) for every tactical video frame.

Input : the feature CSV written by extract_features.py
Output: one row per video frame, columns frame,label  (what possession_metrics.py expects)

Labels: H = home in possession, A = away in possession, D = dead ball, X = abstain.

What the scorer (evaluation/possession_metrics.py) rewards, and how this script uses it:
  * Ranking metric is spell_f1_mean: possessions = maximal runs of H or of A, matched
    one-to-one by temporal IoU (0.3..0.7). Flicker creates false spells, so spells are
    smoothed; a wrong D splits a real possession in two, so D is conservative.
  * X on a graded frame is a frame-level miss, and the temporal metrics forward-fill X
    anyway. So abstaining never helps: X is only written OUTSIDE the graded halves.
  * The file must have exactly one row per frame (0..N-1), so it is never trimmed.

Strategy (rule based, every step is explainable to the domain expert):
    1. Dead ball (D)   : Smart Tagging set-piece / goal frames, ball outside the pitch
                         for a sustained time, ball lost near a boundary. Short alive
                         gaps between two D stretches are bridged into D.
    2. Possession score: a value in [-1, 1] per frame (+1 = home, -1 = away) mixing
                         nearest-player proximity, how long each team has been nearest,
                         whether a team is closing in on the ball, crowding around the
                         ball, and (optional) Smart Tagging team tags as a prior.
    3. Reliability     : proximity evidence is damped when the ball is in flight or far
                         from every player (persistence then carries the possession).
    4. State machine   : first assignment after a stoppage needs |score| >= ENTER_THRESHOLD;
                         switching team needs |score| >= SWITCH_THRESHOLD (hysteresis).
                         In between the previous team is held.
    5. Cleanup         : H/A spells shorter than MIN_SPELL_S are absorbed by their
                         longest neighbour.

Optional modes (need the ground truth CSV, frame,label):
    --gt FILE                    score the prediction with possession_metrics.evaluate
    --tune [--tune-on-half N]    coordinate-descent search of TUNE_GRID on the ground
                                 truth; with --tune-on-half the other half is held out
                                 and reported, to expose over-fitting.

All tunable numbers live in the TUNABLE PARAMETERS block below.
"""

import argparse
import os
import sys
from itertools import groupby

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import DATA_DIR, GAME, GAMES
from starter.baseline import AWAY, HOME

# TUNABLE PARAMETERS  to tweak
# Times are in SECONDS (converted to frames with the game's fps),
# distances in METRES.

# Smart tagging 
USE_SMART_TAGGING = False      # False = tracking features only
DEAD_FROM_SET_PIECE_TAG = True
DEAD_FROM_GOAL_TAG = True
TAG_POSS_BONUS = 0.0         # added to score toward the team in a possession tag
TAG_DEF_BONUS = 0.0          # added toward the OPPONENT of a defending-team tag

# Dead ball (D) 
BALL_OUT_MIN_S = 1.0          # ball outside pitch at least this long -> D
BALL_LOST_DEAD_S = 4.0        # ball untracked this long near a boundary -> D
BOUNDARY_NEAR_M = 3.0         # "near a boundary" distance for the rule above
DEAD_FROM_STATIONARY_BALL = True   # ball (almost) not moving for a while -> D
STATIONARY_SPEED_MS = 1.0     # smoothed ball speed below this = stationary
STATIONARY_MIN_S = 2.0        # ... for at least this long
DEAD_BRIDGE_S = 1.0          # alive gap shorter than this between two D stretches -> D

# Possession score: component weights (relative; normalised internally) 
W_PROXIMITY = 3.0             # nearest-player distance gap (home vs away)
W_PERSISTENCE = 2.0           # share of recent frames each team was nearest
W_TREND = 1.0                 # which team is closing in on the ball
W_CROWD = 1.0                 # players within the near / very-near radii
VERY_NEAR_WEIGHT = 2.0        # extra weight of players within 2 m (crowd term)

# Possession score: scales and reliability 
PROX_SCALE_M = 3.0            # distance gap giving tanh(1) = 0.76 of full evidence
TREND_SCALE_M = 3.0           # change in distance (10 frames) for the same
D_FILL_M = 40.0               # distance used when a team has no tracked player
FAR_FROM_PLAYERS_M = 8.0      # nobody this close -> proximity less reliable
FAR_DAMPING = 0.4             # reliability factor in that case
FLIGHT_SPEED_MS = 12.0        # smoothed ball speed above this = ball in flight
FLIGHT_DAMPING = 0.4          # reliability factor in that case
SCORE_SMOOTH_S = 0.6          # centred rolling-mean window on the score

# Decision rule
ENTER_THRESHOLD = 0.25        # |score| needed to assign a team after a stoppage
SWITCH_THRESHOLD = 0.35       # |score| needed to flip from one team to the other
RESET_AFTER_DEAD = True       # after a D spell, forget the previous possessor
MIN_SPELL_S = 2.0             # absorb H/A spells shorter than this

# Output
SUBMISSION_COLUMNS = ["frame", "label"]   # what read_label_csv requires
UNGRADED_LABEL = "X"                      # written outside the graded halves only
RESULTS_DIR = os.path.join(ROOT, "evaluation", "my_results")   # created if missing

# Persistence feature windows produced by extract_features.py
PERSISTENCE_WINDOWS = (25, 50)
DISTANCE_TREND_LAG = 10

# Tuning search space (used by --tune); keep grids coarse
TUNE_GRID = {
    "ENTER_THRESHOLD": [0.10, 0.20, 0.30, 0.40],
    "SWITCH_THRESHOLD": [0.10, 0.20, 0.30, 0.40, 0.50],
    "MIN_SPELL_S": [0.5, 1.0, 2.0, 3.0, 5.0],
    "SCORE_SMOOTH_S": [0.4, 1.0, 2.0, 3.0],
    "DEAD_BRIDGE_S": [0.5, 1.0, 2.0, 4.0],
    "BALL_OUT_MIN_S": [0.5, 1.0, 2.0, 3.0],
    "W_PROXIMITY": [1.0, 3.0, 5.0],
    "W_PERSISTENCE": [0.0, 1.0, 2.0, 4.0],
    "TAG_POSS_BONUS": [0.0, 0.25, 0.5, 0.75],
}
TUNE_ROUNDS = 2

PARAMETER_NOTES = [
    ("USE_SMART_TAGGING", "tags allowed as input? confirm with organisers"),
    ("DEAD_FROM_SET_PIECE_TAG", "SET PIECES tag -> D"),
    ("DEAD_FROM_GOAL_TAG", "GOAL tag -> D"),
    ("TAG_POSS_BONUS", "score shift toward the team in a possession tag"),
    ("TAG_DEF_BONUS", "score shift away from the team in a defending tag"),
    ("BALL_OUT_MIN_S", "s ball outside pitch -> D"),
    ("BALL_LOST_DEAD_S", "s untracked near a boundary -> D"),
    ("BOUNDARY_NEAR_M", "m, 'near a boundary'"),
    ("DEAD_BRIDGE_S", "s alive gap between two D stretches that becomes D"),
    ("W_PROXIMITY", "weight: nearest-player distance gap"),
    ("W_PERSISTENCE", "weight: share of recent frames each team was nearest"),
    ("W_TREND", "weight: which team is closing in"),
    ("W_CROWD", "weight: players within 5 m / 2 m"),
    ("PROX_SCALE_M", "m, distance-gap saturation"),
    ("FAR_FROM_PLAYERS_M", "m, nobody closer -> proximity damped"),
    ("FLIGHT_SPEED_MS", "m/s, faster ball = in flight -> proximity damped"),
    ("SCORE_SMOOTH_S", "s smoothing of the score"),
    ("ENTER_THRESHOLD", "|score| to assign a team after a stoppage"),
    ("SWITCH_THRESHOLD", "|score| to flip the team in possession"),
    ("RESET_AFTER_DEAD", "forget the possessor after D"),
    ("MIN_SPELL_S", "s minimum H/A spell"),
]


def frames(seconds, fps):
    return max(1, int(round(seconds * fps)))


# helpers
def long_runs(mask, min_len):
    """Keep only the True runs of `mask` that last at least `min_len` frames."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start >= min_len:
            out[start:end] = True
    return out


def fill_short_gaps(mask, max_gap):
    """Set to True the False gaps of at most `max_gap` frames that sit between two True runs."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return out
    gaps = np.diff(idx) - 1
    for k in np.flatnonzero((gaps > 0) & (gaps <= max_gap)):
        out[idx[k] + 1: idx[k] + 1 + gaps[k]] = True
    return out


def near_boundary(x, y, pitch, margin):
    length, width = pitch
    near = (x <= margin) | (x >= length - margin) | (y <= margin) | (y >= width - margin)
    return near & np.isfinite(x) & np.isfinite(y)


def as_bool(series):
    return series.fillna(False).astype(bool).to_numpy()


# dead ball
def dead_sources(table, fps, pitch):
    """Each dead-ball rule as its own boolean mask (before gap bridging)."""
    n = len(table)
    none = np.zeros(n, dtype=bool)

    set_piece = none
    goal = none
    if USE_SMART_TAGGING:
        if DEAD_FROM_SET_PIECE_TAG:
            set_piece = as_bool(table["tag_dead"])
        if DEAD_FROM_GOAL_TAG:
            goal = as_bool(table["tag_goal"])

    out_of_play = long_runs(as_bool(table["ball_out"]), frames(BALL_OUT_MIN_S, fps))

    missing = as_bool(table["ball_missing"])
    since = table["frames_since_ball_seen"].to_numpy()
    last_x = table["last_ball_x"].to_numpy(dtype=float)
    last_y = table["last_ball_y"].to_numpy(dtype=float)
    lost_near_line = (
        missing
        & (since >= frames(BALL_LOST_DEAD_S, fps))
        & near_boundary(last_x, last_y, pitch, BOUNDARY_NEAR_M)
    )
    stationary = none
    if DEAD_FROM_STATIONARY_BALL:
        slow = table["ball_speed_smooth"].fillna(np.inf).to_numpy() < STATIONARY_SPEED_MS
        stationary = long_runs(slow & ~missing, frames(STATIONARY_MIN_S, fps))

    return {
        "set-piece tag": set_piece,
        "goal tag": goal,
        "ball outside pitch": out_of_play,
        "ball lost near line": lost_near_line,
        "ball stationary": stationary,
    }


def dead_frames(table, fps, pitch):
    """Boolean mask of dead-ball frames (D)."""
    dead = np.logical_or.reduce(list(dead_sources(table, fps, pitch).values()))
    return fill_short_gaps(dead, frames(DEAD_BRIDGE_S, fps))


# possession score
def possession_score(table, fps):
    """Score in [-1, 1] per frame: positive favours HOME, negative AWAY."""
    d_home = table["d_home"].fillna(D_FILL_M)
    d_away = table["d_away"].fillna(D_FILL_M)

    proximity = np.tanh((d_away - d_home) / PROX_SCALE_M)

    persistence = sum(
        table["nearest_home_share_{}f".format(w)].fillna(0.0)
        - table["nearest_away_share_{}f".format(w)].fillna(0.0)
        for w in PERSISTENCE_WINDOWS
    ) / len(PERSISTENCE_WINDOWS)

    lag = DISTANCE_TREND_LAG
    trend = np.tanh(
        (
            table["d_away_change_{}f".format(lag)].fillna(0.0)
            - table["d_home_change_{}f".format(lag)].fillna(0.0)
        )
        / TREND_SCALE_M
    )

    home_crowd = table["n_home_near"] + VERY_NEAR_WEIGHT * table["n_home_very_near"]
    away_crowd = table["n_away_near"] + VERY_NEAR_WEIGHT * table["n_away_very_near"]
    total = home_crowd + away_crowd
    crowd = ((home_crowd - away_crowd) / total.where(total > 0)).fillna(0.0)

    reliability = pd.Series(1.0, index=table.index)
    reliability[np.minimum(d_home, d_away) > FAR_FROM_PLAYERS_M] *= FAR_DAMPING
    reliability[table["ball_speed_smooth"].fillna(0.0) > FLIGHT_SPEED_MS] *= FLIGHT_DAMPING

    weights = W_PROXIMITY + W_PERSISTENCE + W_TREND + W_CROWD
    score = (
        W_PROXIMITY * proximity * reliability
        + W_PERSISTENCE * persistence
        + W_TREND * trend * reliability
        + W_CROWD * crowd * reliability
    ) / weights

    if USE_SMART_TAGGING:
        poss = table["tag_team_poss"]
        defn = table["tag_team_def"]
        score = score + TAG_POSS_BONUS * (
            (poss == HOME).astype(float) - (poss == AWAY).astype(float)
        )
        score = score - TAG_DEF_BONUS * (
            (defn == HOME).astype(float) - (defn == AWAY).astype(float)
        )

    score = score.clip(-1.0, 1.0)
    window = frames(SCORE_SMOOTH_S, fps)
    smoothed = score.groupby(table["half"]).transform(
        lambda s: s.rolling(window, center=True, min_periods=1).mean()
    )
    return smoothed.to_numpy(dtype=float)


# state machine
def label_half(score, dead):
    """H/A/D labels for one half. Never abstains: X would only be a frame-level miss."""
    labels = np.empty(len(score), dtype="<U1")
    state = None

    for i in range(len(score)):
        if dead[i]:
            labels[i] = "D"
            if RESET_AFTER_DEAD:
                state = None
            continue

        s = score[i]
        if state is None:
            if s >= ENTER_THRESHOLD:
                state = "H"
            elif s <= -ENTER_THRESHOLD:
                state = "A"
        elif state == "A" and s >= SWITCH_THRESHOLD:
            state = "H"
        elif state == "H" and s <= -SWITCH_THRESHOLD:
            state = "A"

        # No team established yet (start of half / just after a stoppage): follow the sign.
        labels[i] = state if state else ("H" if s >= 0 else "A")
    return labels


def merge_short_spells(labels, min_len):
    """Absorb H/A spells shorter than min_len into their longest alive neighbour."""
    runs = [[lab, len(list(grp))] for lab, grp in groupby(labels)]

    while True:
        best = None
        for k, (lab, length) in enumerate(runs):
            if lab == "D" or length >= min_len:
                continue
            left = runs[k - 1] if k > 0 and runs[k - 1][0] != "D" else None
            right = runs[k + 1] if k + 1 < len(runs) and runs[k + 1][0] != "D" else None
            if left is None and right is None:
                continue
            if best is None or length < best[0]:
                best = (length, k, left, right)
        if best is None:
            break

        _, k, left, right = best
        if left is None:
            target = right
        elif right is None:
            target = left
        else:
            target = left if left[1] >= right[1] else right
        runs[k][0] = target[0]

        merged = [runs[0]]
        for run in runs[1:]:
            if run[0] == merged[-1][0]:
                merged[-1][1] += run[1]
            else:
                merged.append(run)
        runs = merged

    return np.repeat([r[0] for r in runs], [r[1] for r in runs]).astype("<U1")


# pipeline
def predict(table, game):
    fps = GAMES[game]["fps"]
    pitch = GAMES[game]["pitch"]

    dead = dead_frames(table, fps, pitch)
    score = possession_score(table, fps)

    labels = np.full(len(table), UNGRADED_LABEL, dtype="<U1")
    half = table["half"].to_numpy()
    min_len = frames(MIN_SPELL_S, fps)

    for number in sorted(set(half.tolist()) - {0}):
        idx = np.flatnonzero(half == number)
        labels[idx] = merge_short_spells(label_half(score[idx], dead[idx]), min_len)

    out = pd.DataFrame({"frame": table["frame"].to_numpy(), "label": labels})
    out["half"] = half
    out["score"] = score
    return out


# scoring and validation
def load_metrics():
    try:
        import possession_metrics as pm
    except ImportError:
        return None
    return pm


def validate(out, n_expected, pm):
    """Return None if the submission is valid, else the reason (same check as the scorer)."""
    frame = out[SUBMISSION_COLUMNS]
    if pm is not None:
        return pm.validate_submission(frame, n_expected)
    if len(frame) != n_expected or not np.array_equal(frame["frame"].to_numpy(),
                                                      np.arange(n_expected)):
        return "frame column must be 0..{} in order".format(n_expected - 1)
    return None


def restrict_to_half(gt_codes, half, number, pm):
    """Copy of the ground-truth codes with every frame outside half `number` set to X."""
    restricted = gt_codes.copy()
    restricted[half != number] = pm.X
    return restricted


def objective(result):
    """Ranking order of the scorer: spell_f1_mean first, macro_f1 as tie-break."""
    return (round(result["spell_f1_mean"], 4), round(result["macro_f1"], 4))


def print_scores(result, title):
    print("\n--- {} ---".format(title))
    print("  spell_f1_mean {:.4f}   (IoU 0.5: {:.4f}; noise band about +/-0.04)".format(
        result["spell_f1_mean"], result["spell_f1_at_headline"]))
    print("  macro_f1      {:.4f}   alive/dead F1 {:.4f}   team acc (both alive) {:.3f}".format(
        result["macro_f1"], result["alive_dead_f1"], result["team_acc_both_alive"]))
    for name, stats in result["per_class"].items():
        print("  {}: precision {:.3f} recall {:.3f} f1 {:.3f} (support {})".format(
            name, stats["precision"], stats["recall"], stats["f1"], stats["support"]))
    print("  possessions: predicted {} vs ground truth {}; mean length {:.1f}s vs {:.1f}s".format(
        result["n_pred_spells"], result["n_gt_spells"],
        result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]))
    print("  abstentions on graded frames: {}".format(result["n_abstain_on_graded"]))


# tuning
def set_param(name, value):
    globals()[name] = value


def tune(table, game, gt_codes, pm):
    """Coordinate descent over TUNE_GRID maximising the scorer's ranking order."""
    def run(codes):
        out = predict(table, game)
        return pm.evaluate(codes, pm.label_codes(out["label"].to_numpy()))

    best = objective(run(gt_codes))
    print("start: spell_f1_mean {:.4f}, macro_f1 {:.4f}".format(*best))
    for round_no in range(TUNE_ROUNDS):
        improved = False
        for name, grid in TUNE_GRID.items():
            original = globals()[name]
            best_value = original
            for value in grid:
                set_param(name, value)
                score = objective(run(gt_codes))
                if score > best:
                    best, best_value, improved = score, value, True
            set_param(name, best_value)
            print("  round {} {:<18} -> {}  (spell_f1_mean {:.4f}, macro_f1 {:.4f})".format(
                round_no + 1, name, best_value, *best))
        if not improved:
            break
    return best


# diagnosis
ABLATIONS = [
    ("current settings", {}),
    ("no Smart Tagging at all", {"USE_SMART_TAGGING": False}),
    ("tags for D only (no team prior)", {"TAG_POSS_BONUS": 0.0, "TAG_DEF_BONUS": 0.0}),
    ("no set-piece D", {"DEAD_FROM_SET_PIECE_TAG": False}),
    ("no D at all (tags kept for team)", {
        "DEAD_FROM_SET_PIECE_TAG": False, "DEAD_FROM_GOAL_TAG": False,
        "BALL_OUT_MIN_S": 1e6, "BALL_LOST_DEAD_S": 1e6}),
    ("tracking only, no D", {
        "USE_SMART_TAGGING": False, "BALL_OUT_MIN_S": 1e6, "BALL_LOST_DEAD_S": 1e6}),
    ("longer spells (switch 0.4, min 3 s)", {"SWITCH_THRESHOLD": 0.4, "MIN_SPELL_S": 3.0}),
]


def mask_runs(mask):
    padded = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def label_mix(codes, pm):
    counts = np.bincount(codes, minlength=pm.N_LABELS)
    total = max(1, counts.sum())
    return "  ".join("{} {:3.0f}%".format(pm.LABELS[k], 100.0 * counts[k] / total)
                     for k in range(pm.N_LABELS) if counts[k])


def diagnose(table, game, gt_codes, pm, n_segments=15):
    """Show where the prediction disagrees with the ground truth and which rule causes it."""
    fps = GAMES[game]["fps"]
    pitch = GAMES[game]["pitch"]
    out = predict(table, game)
    pred = pm.label_codes(out["label"].to_numpy())
    result = pm.evaluate(gt_codes, pred)
    runs = pm.valid_runs(gt_codes)

    print("\n=== DIAGNOSIS ===")
    print("ground-truth graded frames : {}".format(runs))
    print("this script's graded halves: {}".format(mask_runs(table["half"].to_numpy() > 0)))

    print("\nconfusion (rows = ground truth, columns = prediction), frames:")
    print("          " + "".join("{:>9}".format(name) for name in pm.LABELS))
    for k, name in enumerate(pm.SCORED):
        print("  GT {:<3} ".format(name) + "".join("{:>9}".format(v) for v in result["confusion"][k]))
    team = result["team_acc_both_alive"]
    print("team accuracy when both say the ball is alive: {:.1%}".format(team))
    if team < 0.5:
        print("  WARNING: below 50% -> check whether HOME/AWAY are swapped (tags or tracks)")

    print("\nfirst {} ground-truth segments and what we predicted inside them:".format(n_segments))
    for start, stop, code in pm.segments(gt_codes, runs)[:n_segments]:
        print("  frames {:>7}-{:<7} {:5.1f}s  GT {}  ->  {}".format(
            start, stop, (stop - start) / float(fps), pm.LABELS[code],
            label_mix(pred[start:stop], pm)))

    graded = gt_codes != pm.X
    sources = dead_sources(table, fps, pitch)
    union = np.logical_or.reduce(list(sources.values()))
    sources["added by bridging"] = dead_frames(table, fps, pitch) & ~union
    print("\nD rules inside the graded frames (share of those frames the ground truth calls D):")
    for name, mask in sources.items():
        selected = mask & graded
        if selected.any():
            print("  {:<22} {:>7} frames, {:5.1%} are GT D".format(
                name, int(selected.sum()), float(np.mean(gt_codes[selected] == pm.D))))
        else:
            print("  {:<22} {:>7} frames".format(name, 0))

    print("\nablations on the full ground truth:")
    print("  {:<36} {:>8} {:>8} {:>6} {:>8} {:>10}".format(
        "variant", "spellF1", "macroF1", "D-F1", "teamAcc", "poss p/gt"))
    for name, overrides in ABLATIONS:
        saved = {key: globals()[key] for key in overrides}
        for key, value in overrides.items():
            set_param(key, value)
        try:
            res = pm.evaluate(gt_codes, pm.label_codes(predict(table, game)["label"].to_numpy()))
        finally:
            for key, value in saved.items():
                set_param(key, value)
        print("  {:<36} {:>8.4f} {:>8.4f} {:>6.3f} {:>8.3f} {:>6}/{}".format(
            name, res["spell_f1_mean"], res["macro_f1"], res["per_class"]["D"]["f1"],
            res["team_acc_both_alive"], res["n_pred_spells"], res["n_gt_spells"]))


def print_best_parameters():
    print("\n=== TUNED VALUES (paste into TUNABLE PARAMETERS) ===")
    for name in TUNE_GRID:
        print("{} = {}".format(name, globals()[name]))


# reports
def print_summary(out, features):
    print("\n=== STRATEGY ===")
    print("Strategy" + __doc__.split("Strategy", 1)[1].split("Optional modes", 1)[0].rstrip())

    print("\n=== THRESHOLDS (edit in TUNABLE PARAMETERS) ===")
    for name, note in PARAMETER_NOTES:
        print("  {:<26} {:<8} {}".format(name, str(globals()[name]), note))

    graded = out["half"] > 0
    g = out[graded]
    print("\n=== PREDICTION SUMMARY ({} graded frames) ===".format(len(g)))
    pred = g["label"].to_numpy()
    spells = {}
    for lab, _ in groupby(pred):
        spells[lab] = spells.get(lab, 0) + 1
    for lab, count in g["label"].value_counts().items():
        print("  {}: {:6.1%} of frames, {} spells".format(lab, count / len(g), spells[lab]))

    nearest = features.loc[graded.to_numpy(), "nearest_team"].to_numpy()
    both = np.isin(pred, ["H", "A"]) & np.isin(nearest, [HOME, AWAY])
    if both.any():
        agree = np.mean((pred[both] == "H") == (nearest[both] == HOME))
        print("  agreement with plain nearest-team on H/A frames: {:.1%}".format(agree))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--features", help="feature CSV (default: derived/<game>_features.csv)")
    parser.add_argument("--out", help="output CSV (default: evaluation/my_results/<game>_predictions.csv)")
    parser.add_argument("--gt", help="ground-truth CSV (frame,label) to score against")
    parser.add_argument("--tune", action="store_true", help="search TUNE_GRID on --gt")
    parser.add_argument("--tune-on-half", type=int, choices=(1, 2),
                        help="tune on this half only and report the other half held out")
    parser.add_argument("--diagnose", action="store_true",
                        help="with --gt: confusion matrix, segment-by-segment view, D rules, ablations")
    parser.add_argument("--debug", action="store_true", help="also write half and score")
    args = parser.parse_args()

    features_path = args.features or os.path.join(
        DATA_DIR, "derived", "{}_features.csv".format(args.game)
    )
    out_path = args.out or os.path.join(
        RESULTS_DIR, "{}_predictions.csv".format(args.game)
    )

    features = pd.read_csv(features_path)
    pm = load_metrics()

    gt_codes = None
    if args.gt:
        if pm is None:
            sys.exit("cannot import possession_metrics (expected in evaluation/)")
        gt_codes = pm.label_codes(pm.read_label_csv(args.gt))
        if len(gt_codes) != len(features):
            sys.exit("ground truth has {} rows, features {}".format(len(gt_codes), len(features)))

    half = features["half"].to_numpy()
    if args.tune:
        if gt_codes is None:
            sys.exit("--tune needs --gt")
        tune_codes = (restrict_to_half(gt_codes, half, args.tune_on_half, pm)
                      if args.tune_on_half else gt_codes)
        tune(features, args.game, tune_codes, pm)
        print_best_parameters()

    out = predict(features, args.game)

    reason = validate(out, len(features), pm)
    if reason:
        sys.exit("invalid submission: {}".format(reason))

    columns = SUBMISSION_COLUMNS + (["half", "score"] if args.debug else [])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out[columns].to_csv(out_path, index=False)
    print("Wrote {} ({} rows)".format(out_path, len(out)))

    if gt_codes is not None:
        pred_codes = pm.label_codes(out["label"].to_numpy())
        if args.tune_on_half:
            other = 2 if args.tune_on_half == 1 else 1
            print_scores(pm.evaluate(restrict_to_half(gt_codes, half, args.tune_on_half, pm),
                                     pred_codes), "tuned half {}".format(args.tune_on_half))
            print_scores(pm.evaluate(restrict_to_half(gt_codes, half, other, pm), pred_codes),
                         "HELD-OUT half {}".format(other))
        else:
            print_scores(pm.evaluate(gt_codes, pred_codes), "full match")

    if args.diagnose:
        if gt_codes is None:
            sys.exit("--diagnose needs --gt")
        diagnose(features, args.game, gt_codes, pm)

    print_summary(out, features)


if __name__ == "__main__":
    main()