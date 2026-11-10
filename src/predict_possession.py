"""Predict possession labels (H / A / D) for every tactical video frame.

Input : the feature CSV written by extract_features.py
Output: one row per video frame, columns frame,label  (what possession_metrics.py expects)

Labels: H = home in possession, A = away in possession, D = dead ball, X = abstain.

This script does NOT read any ground truth and has no tuning or diagnosis mode.
Every number is set by hand in the PARAMETERS block below, to be reviewed with the
football expert (see parameters_for_expert.md).

What the scorer (evaluation/possession_metrics.py) rewards, and how this script uses it:
  * Ranking metric is spell_f1_mean: possessions = maximal runs of H or of A, matched
    one-to-one by temporal IoU (0.3..0.7). Flicker creates false spells, so spells are
    smoothed; a wrong D splits a real possession in two, so D is conservative.
  * X on a graded frame is a frame-level miss, and the temporal metrics forward-fill X
    anyway. So abstaining never helps: X is only written OUTSIDE the graded halves.
  * The file must have exactly one row per frame (0..N-1), so it is never trimmed.

Strategy (rule based, every step is explainable to the domain expert):
    1. Dead ball (D)   : ball outside the pitch for a sustained time, ball lost near a
                         boundary, ball not moving for a while (optionally Smart Tagging
                         set-piece / goal frames). Short alive gaps between two D
                         stretches are bridged into D. Graded frames before the
                         second-half kick-off are D (FAQ; frames from games.py).
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

    With USE_HMM, steps 4-5 (and the score smoothing and D gap bridging) are replaced by a
    three-state hidden Markov model decoded with Viterbi: evidence = proximity x reliability
    for H/A and the dead-ball rules as soft probabilities for D; transition probabilities
    come from published football statistics (see the HMM block of the PARAMETERS).

All tunable numbers live in the PARAMETERS block below.
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

from games import DATA_DIR, GAME, GAMES, anchor_video_frame, half_video_blocks
from starter.baseline import AWAY, HOME


# Use of Smart Tagging 
USE_SMART_TAGGING = False     # [CLIP] False = tracking features only
DEAD_FROM_SET_PIECE_TAG = True   # only used when USE_SMART_TAGGING is True
DEAD_FROM_GOAL_TAG = True        # only used when USE_SMART_TAGGING is True
TAG_POSS_BONUS = 0.0          # [CLIP] added to score toward the team in a possession tag
TAG_DEF_BONUS = 0.0           # [CLIP] added toward the OPPONENT of a defending-team tag

# Dead ball (D)
BALL_OUT_MIN_S = 1.0          # ball outside pitch at least this long -> D
BALL_LOST_DEAD_S = 4.0        # ball untracked this long near a boundary -> D
BOUNDARY_NEAR_M = 3.0         # "near a boundary" distance for the rule above
DEAD_FROM_STATIONARY_BALL = True   # [CLIP] rule added after seeing D-rule results
STATIONARY_SPEED_MS = 1.0     # [CLIP] smoothed ball speed below this = stationary
STATIONARY_MIN_S = 2.0        # [CLIP] ... for at least this long
DEAD_BRIDGE_S = 1.0           # alive gap shorter than this between two D stretches -> D
D_BEFORE_KICKOFF = True       # FAQ: graded frames before the 2nd-half kick-off are all D
PRE_KICKOFF_ANCHORS = {2: "kickoff_2nd"}   # half -> kick-off anchor in games.py; half 1 is graded from its kick-off

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
SWITCH_THRESHOLD = 0.35       # [CLIP] |score| needed to flip from one team to the other
RESET_AFTER_DEAD = True       # after a D spell, forget the previous possessor
MIN_SPELL_S = 2.0             # [CLIP] absorb H/A spells shorter than this

# Hidden Markov model + Viterbi. When on, it replaces the score smoothing, the state machine,
# the D gap bridging and the spell cleanup above. Evidence: proximity x reliability for H/A,
# the dead-ball rules as soft probabilities for D.
USE_HMM = True
MEAN_POSSESSION_S = 15.0      # ~57 min ball in play / ~230 possessions (Taha & Ali 2023; Jerome et al. 2024)
MEAN_DEAD_S = 18.7            # mean game interruption (Siegle & Lames 2012)
SHARE_ENDED_BY_STOPPAGE = 0.47  # running phase 32.1 s / possession 15 s (Siegle & Lames 2012)
MIN_CONTROL_S = 2.0           # CHALLENGE.md: possession only changes on controlled possession ("a player
                              # keeps the ball for a few seconds"), so a possession won in open play lasts at least this
FLIP_SCORE = 0.5              # counter-evidence of this strength ...
FLIP_S = 1.0                  # ... must last this long to start a new possession (one player possession ~1.1 s, Carling 2010)
DEAD_PROB_BASE = 0.1          # dead-ball probability on a frame where no rule fires
DEAD_PROB = {                 # dead-ball probability when a rule fires (combined by noisy-OR)
    "ball outside pitch": 0.95,   # IFAB Law 9: out of play once wholly over the line
    "ball lost near line": 0.7,
    "ball stationary": 0.6,       # IFAB Laws 13/16: ball stationary at restarts, but also when held
    "set-piece tag": 0.6,
    "goal tag": 0.9,
    "before kick-off": 0.999,     # FAQ: all D
}

# Output
SUBMISSION_COLUMNS = ["frame", "label"]   # what read_label_csv requires
UNGRADED_LABEL = "X"                      # written outside the graded halves only
RESULTS_DIR = os.path.join(ROOT, "evaluation", "my_results")   # created if missing

# Feature windows produced by extract_features.py (must match that file)
PERSISTENCE_WINDOWS = (25, 50)
DISTANCE_TREND_LAG = 10
# =============================================================================

PARAMETER_NOTES = [
    ("USE_SMART_TAGGING", "tags used as input?"),
    ("DEAD_FROM_SET_PIECE_TAG", "SET PIECES tag -> D (needs USE_SMART_TAGGING)"),
    ("DEAD_FROM_GOAL_TAG", "GOAL tag -> D (needs USE_SMART_TAGGING)"),
    ("TAG_POSS_BONUS", "score shift toward the team in a possession tag"),
    ("TAG_DEF_BONUS", "score shift away from the team in a defending tag"),
    ("BALL_OUT_MIN_S", "s ball outside pitch -> D"),
    ("BALL_LOST_DEAD_S", "s untracked near a boundary -> D"),
    ("BOUNDARY_NEAR_M", "m, 'near a boundary'"),
    ("DEAD_FROM_STATIONARY_BALL", "stationary ball -> D"),
    ("STATIONARY_SPEED_MS", "m/s, below this the ball counts as stationary"),
    ("STATIONARY_MIN_S", "s the ball must stay stationary"),
    ("DEAD_BRIDGE_S", "s alive gap between two D stretches that becomes D"),
    ("D_BEFORE_KICKOFF", "graded frames before the 2nd-half kick-off -> D"),
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
    ("USE_HMM", "HMM + Viterbi instead of smoothing/state machine/cleanup"),
    ("MEAN_POSSESSION_S", "s, HMM prior: mean possession"),
    ("MEAN_DEAD_S", "s, HMM prior: mean stoppage"),
    ("SHARE_ENDED_BY_STOPPAGE", "HMM prior: share of possessions ended by a stoppage"),
    ("MIN_CONTROL_S", "s a possession won in open play must last (controlled possession)"),
    ("FLIP_SCORE", "HMM: counter-evidence strength ..."),
    ("FLIP_S", "s ... needed to start a new possession"),
    ("DEAD_PROB_BASE", "HMM: dead-ball probability when no rule fires"),
]


def frames(seconds, fps):
    return max(1, int(round(seconds * fps)))


# helpers
def long_runs(mask, min_len):
    """Keep only the True runs of mask that last at least min_len frames."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start >= min_len:
            out[start:end] = True
    return out


def fill_short_gaps(mask, max_gap):
    """Set to True the False gaps of at most max_gap frames that sit between two True runs."""
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


def pre_kickoff_frames(table, game):
    """Graded frames before their half's kick-off, which the FAQ defines as D.

    The second half is graded from the first frame of its tracking block (110,808) but the ball
    is struck at the kick-off anchor (113,741): those 2,934 frames, anchor included, are D. Both
    frames come from games.py, so nothing here is read off the video or the ground truth.
    """
    frame = table["frame"].to_numpy()
    mask = np.zeros(len(table), dtype=bool)
    first_graded = {half: first for half, first, _ in half_video_blocks(game)}
    for half, anchor in PRE_KICKOFF_ANCHORS.items():
        kickoff = anchor_video_frame(GAMES[game][anchor])
        mask |= (frame >= first_graded[half]) & (frame <= kickoff)
    return mask


# possession score
def proximity_evidence(table):
    """Nearest-player distance gap in [-1, 1]: positive when the nearest home player is closer."""
    d_home = table["d_home"].fillna(D_FILL_M)
    d_away = table["d_away"].fillna(D_FILL_M)
    return np.tanh((d_away - d_home) / PROX_SCALE_M)


def reliability_factor(table):
    """1.0 where proximity says something about possession, damped for a ball in flight or far from everyone."""
    nearest = np.minimum(table["d_home"].fillna(D_FILL_M), table["d_away"].fillna(D_FILL_M))
    reliability = pd.Series(1.0, index=table.index)
    reliability[nearest > FAR_FROM_PLAYERS_M] *= FAR_DAMPING
    reliability[table["ball_speed_smooth"].fillna(0.0) > FLIGHT_SPEED_MS] *= FLIGHT_DAMPING
    return reliability


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

    reliability = reliability_factor(table)

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


# hidden Markov model
HMM_STATES = np.array(["H", "A", "D"])


def hmm_transitions(fps):
    """Per-frame log transition matrix [from, to] over H, A, D, built from the football priors.

    A possession's first MIN_CONTROL_S are forced by the duration chain in viterbi(), so only the
    remainder of the mean possession is left to the geometric "steady" part modelled here.
    """
    p_end = 1 / ((MEAN_POSSESSION_S - MIN_CONTROL_S) * fps)  # a possession ends, either way
    p_go_dead = p_end * SHARE_ENDED_BY_STOPPAGE       # ...by a stoppage
    p_switch = p_end * (1 - SHARE_ENDED_BY_STOPPAGE)  # ...by a turnover in open play
    p_dead_end = 1 / (MEAN_DEAD_S * fps)
    T = np.array([
        [1 - p_end, p_switch, p_go_dead],                  # from H
        [p_switch, 1 - p_end, p_go_dead],                  # from A
        [p_dead_end / 2, p_dead_end / 2, 1 - p_dead_end],  # from D: either team restarts
    ])
    return np.log(T)


def evidence_gain(log_T, fps):
    """Scale of the H/A evidence, derived from FLIP_SCORE / FLIP_S instead of being set by hand.

    A new possession inside another one costs two switches (there and back). It should appear once
    counter-evidence of strength FLIP_SCORE has lasted FLIP_S, and H and A differ by 2 * gain * score
    per frame, so gain = 2 * switch_cost / (2 * FLIP_SCORE * frames).
    """
    switch_cost = log_T[0, 0] - log_T[0, 1]
    return switch_cost / (FLIP_SCORE * frames(FLIP_S, fps))


def dead_probability(sources):
    """Per-frame dead-ball probability: noisy-OR of the rules that fire, on top of DEAD_PROB_BASE."""
    alive = np.full(len(next(iter(sources.values()))), 1 - DEAD_PROB_BASE)
    for name, mask in sources.items():
        alive = alive * np.where(mask, 1 - DEAD_PROB[name], 1.0)
    return 1 - alive


def hmm_log_emissions(evidence, dead_prob, gain):
    """(n, 3) log-emissions for H, A, D. evidence in [-1, 1] (+ = home), already damped by reliability."""
    d = np.clip(dead_prob, 1e-3, 1 - 1e-3)
    alive = np.log1p(-d)
    s = np.nan_to_num(evidence)
    return np.column_stack([gain * s + alive, -gain * s + alive, np.log(d)])


def viterbi(log_e, log_T, min_len):
    """Most likely state sequence (indices into HMM_STATES) for one half.

    A possession won in open play must last at least min_len frames (CHALLENGE.md: only a
    controlled possession changes hands). Each team has a chain of min_len states: a turnover
    enters the opponent's first chain state and must walk the whole chain, so shorter spells are
    impossible. A restart after D enters the last ("steady") state directly, because the rules give
    the restart team the ball. Chain states cannot go dead: if the ball goes out before the
    challenger has control, the best path keeps the previous team until the stoppage.
    """
    m = max(1, min_len)
    stay, switch, go_dead = log_T[0, 0], log_T[0, 1], log_T[0, 2]
    restart, dead_stay = log_T[2, 0], log_T[2, 2]
    n = len(log_e)

    # v[team] holds the chain scores, index m - 1 being the steady state; team 0 = H, 1 = A.
    v = np.full((2, m), -np.inf)
    v[0, m - 1], v[1, m - 1] = log_e[0, 0], log_e[0, 1]   # a half starts with a kick-off: no chain
    v_dead = log_e[0, 2]
    # Back-pointers, stored only where there is a choice: 0 = stayed, 1 = came from the chain
    # (or from the other team's steady state when m == 1), 2 = came from D.
    back_steady = np.zeros((n, 2), dtype=np.int8)
    back_dead = np.zeros(n, dtype=np.int8)      # 0 = stayed dead, 1 = from H, 2 = from A

    for t in range(1, n):
        new = np.empty_like(v)
        for team in (0, 1):
            other = v[1 - team, m - 1]
            own = v[team]
            before = own[m - 2] if m > 1 else other + switch
            cand = (own[m - 1] + stay, before, v_dead + restart)
            k = int(np.argmax(cand))
            back_steady[t, team] = k
            new[team, m - 1] = cand[k]
            if m > 1:
                new[team, 1:m - 1] = own[0:m - 2]
                new[team, 0] = other + switch
        cand = (v_dead + dead_stay, v[0, m - 1] + go_dead, v[1, m - 1] + go_dead)
        k = int(np.argmax(cand))
        back_dead[t] = k
        v_dead = cand[k] + log_e[t, 2]
        new[0] += log_e[t, 0]
        new[1] += log_e[t, 1]
        v = new

    # Backtrack over (team, chain index); team 2 = D.
    best = int(np.argmax(v.ravel()))
    state = (2, 0) if v_dead >= v.ravel()[best] else divmod(best, m)
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        team, k = state
        path[t] = team
        if t == 0:
            break
        if team == 2:
            state = [(2, 0), (0, m - 1), (1, m - 1)][back_dead[t]]
        elif k == m - 1:
            came = back_steady[t, team]
            if came == 0:
                state = (team, m - 1)
            elif came == 1:
                state = (team, m - 2) if m > 1 else (1 - team, m - 1)
            else:
                state = (2, 0)
        elif k > 0:
            state = (team, k - 1)
        else:
            state = (1 - team, m - 1)
    return path


def predict_hmm(table, game):
    """H/A/D labels from the HMM, X outside the graded halves."""
    fps = GAMES[game]["fps"]
    pitch = GAMES[game]["pitch"]

    sources = dict(dead_sources(table, fps, pitch))
    if D_BEFORE_KICKOFF:
        sources["before kick-off"] = pre_kickoff_frames(table, game)
    dead_prob = dead_probability(sources)

    evidence = (proximity_evidence(table) * reliability_factor(table)).to_numpy(dtype=float)
    log_T = hmm_transitions(fps)
    log_e = hmm_log_emissions(evidence, dead_prob, evidence_gain(log_T, fps))

    labels = np.full(len(table), UNGRADED_LABEL, dtype="<U1")
    half = table["half"].to_numpy()
    for number in sorted(set(half.tolist()) - {0}):
        idx = np.flatnonzero(half == number)
        labels[idx] = HMM_STATES[viterbi(log_e[idx], log_T, frames(MIN_CONTROL_S, fps))]

    out = pd.DataFrame({"frame": table["frame"].to_numpy(), "label": labels})
    out["half"] = half
    out["score"] = evidence
    return out


# pipeline
def predict(table, game):
    if USE_HMM:
        return predict_hmm(table, game)
    fps = GAMES[game]["fps"]
    pitch = GAMES[game]["pitch"]

    dead = dead_frames(table, fps, pitch)
    if D_BEFORE_KICKOFF:
        dead |= pre_kickoff_frames(table, game)
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


# format validation
def validate(out, n_expected):
    """Return None if the file format is valid, else the reason. Uses no ground truth."""
    frame = out[SUBMISSION_COLUMNS]
    try:
        import possession_metrics as pm
    except ImportError:
        pm = None
    if pm is not None:
        return pm.validate_submission(frame, n_expected)
    if len(frame) != n_expected or not np.array_equal(frame["frame"].to_numpy(),
                                                      np.arange(n_expected)):
        return "frame column must be 0..{} in order".format(n_expected - 1)
    return None


# reports
def print_summary(out, features):
    print("\n=== STRATEGY ===")
    print("Strategy" + __doc__.split("Strategy", 1)[1].split("All tunable", 1)[0].rstrip())

    print("\n=== PARAMETERS (edit in the PARAMETERS block) ===")
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
    parser.add_argument("--debug", action="store_true", help="also write half and score")
    args = parser.parse_args()

    features_path = args.features or os.path.join(
        DATA_DIR, "derived", "{}_features.csv".format(args.game)
    )
    out_path = args.out or os.path.join(
        RESULTS_DIR, "{}_predictions.csv".format(args.game)
    )

    features = pd.read_csv(features_path)
    out = predict(features, args.game)

    reason = validate(out, len(features))
    if reason:
        sys.exit("invalid submission: {}".format(reason))

    columns = SUBMISSION_COLUMNS + (["half", "score"] if args.debug else [])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out[columns].to_csv(out_path, index=False)
    print("Wrote {} ({} rows)".format(out_path, len(out)))

    print_summary(out, features)


if __name__ == "__main__":
    main()