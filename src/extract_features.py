"""Build frame-level features from ATD tracking and Smart Tagging.

The output is one row per tactical video frame. Coordinates and distances are in metres.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import (
    DATA_DIR,
    GAME,
    GAMES,
    half_video_blocks,
    require,
    source_frames,
)
from starter.baseline import (  
    AWAY,
    HOME,
    NOBODY,
    nearest_player_labels,
    team_names,
    team_of_each_track,
)
from starter import load_smart_tagging 


NEAR_RADIUS_M = 5.0
VERY_NEAR_RADIUS_M = 2.0
OUT_MARGIN_M = 0.5
SPEED_SMOOTH_FRAMES = 5
DISTANCE_TREND_LAG = 10
PERSISTENCE_WINDOWS = (25, 50)

OUT_PATH = os.path.join(DATA_DIR, "derived", "{}_features.{}")

FIELD_SEP = ":"
GROUP_SEP = ";"

SET_PIECE_CODE = "SET PIECES"
GOAL_CODE = "GOAL"
DEFENDING_CODES = {
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}

TAG_TEAMS = {
    "barcelona": HOME,
    "athletic club": AWAY,
    "atletic club": AWAY,
}
TAG_NO_TEAM = {"", "N/A", "NA", "NONE", "UNKNOWN"}


# ATD tracking

def read_atd(raw_path, n_frames, n_slots, pitch):
    """Read normalised ATD x/y coordinates and scale them to pitch metres.

    Each player group contains x,y,speed. Only x/y are needed for the spatial
    features below. Missing coordinates are represented by NaN.
    """
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame = int(parts[0]) - 1  # ATD frame IDs start at 1.
            if not 0 <= frame < n_frames:
                continue

            player_groups = parts[1].strip(GROUP_SEP).split(GROUP_SEP)
            if len(player_groups) != n_slots:
                raise ValueError(
                    "frame_id {}: {} player slots, expected {}".format(
                        parts[0], len(player_groups), n_slots
                    )
                )

            for slot, group in enumerate(player_groups):
                values = group.split(",")
                if len(values) < 2 or values[0].startswith("N"):
                    continue
                players[frame, slot] = (
                    float(values[0]) * length,
                    float(values[1]) * width,
                )

            ball_values = parts[2].strip(GROUP_SEP).split(",")
            if len(ball_values) >= 2 and not ball_values[0].startswith("N"):
                ball[frame] = (
                    float(ball_values[0]) * length,
                    float(ball_values[1]) * width,
                )

    return players, ball


def nearest_by_team(players, ball, team_of_track):
    """Per-frame nearest distance and player counts for each team."""
    distance = np.hypot(
        players[..., 0] - ball[:, None, 0],
        players[..., 1] - ball[:, None, 1],
    )
    tracked = np.isfinite(players[..., 0]) & np.isfinite(players[..., 1])

    out = {}
    for name, team in (("home", HOME), ("away", AWAY)):
        team_slots = team_of_track == team
        team_distances = distance[:, team_slots]
        team_tracked = tracked[:, team_slots]

        # fmin ignores NaNs; rows with no tracked player remain NaN.
        out["d_" + name] = np.fmin.reduce(team_distances, axis=1).astype(
            np.float32
        )
        out["n_{}_near".format(name)] = np.sum(
            team_distances < NEAR_RADIUS_M, axis=1
        ).astype(np.int16)
        out["n_{}_very_near".format(name)] = np.sum(
            team_distances < VERY_NEAR_RADIUS_M, axis=1
        ).astype(np.int16)
        out["n_{}_tracked".format(name)] = np.sum(
            team_tracked, axis=1
        ).astype(np.int16)

    return out


def nearest_team(d_home, d_away):
    """Team of the nearest player, using the baseline's HOME tie-break."""
    has_home = np.isfinite(d_home)
    has_away = np.isfinite(d_away)
    labels = np.full(len(d_home), NOBODY, dtype=np.int8)

    labels[has_home & (~has_away | (d_home <= d_away))] = HOME
    labels[has_away & (~has_home | (d_away < d_home))] = AWAY
    return labels


# Temporal and ball-motion features
def ball_motion(ball, fps):
    """Ball speed, smoothed speed and acceleration; missing positions stay NaN."""
    step = np.full(len(ball), np.nan, dtype=np.float32)
    step[1:] = np.hypot(*np.diff(ball, axis=0).T)

    speed = step * fps
    smooth_speed = (
        pd.Series(speed)
        .rolling(SPEED_SMOOTH_FRAMES, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float32)
    )

    acceleration = np.full(len(ball), np.nan, dtype=np.float32)
    acceleration[1:] = np.diff(smooth_speed) * fps
    return speed, smooth_speed, acceleration


def last_seen(ball):
    """Frames since last visible ball position, plus its last known x/y."""
    index = np.arange(len(ball))
    seen = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    last_index = np.maximum.accumulate(np.where(seen, index, -1))
    known = last_index >= 0

    frames_since = np.full(len(ball), -1, dtype=np.int32)
    frames_since[known] = index[known] - last_index[known]

    last_position = np.full_like(ball, np.nan)
    last_position[known] = ball[last_index[known]]
    return frames_since, last_position[:, 0], last_position[:, 1]


def add_temporal_features(table):
    """Add short-term distance trends and trailing nearest-team persistence."""
    for team in ("home", "away"):
        distance = table["d_" + team]
        table["d_{}_change_{}f".format(team, DISTANCE_TREND_LAG)] = (
            distance - distance.shift(DISTANCE_TREND_LAG)
        )

    nearest = table["nearest_team"]
    for window in PERSISTENCE_WINDOWS:
        # Trailing proportions use only current and previous frames
        for team, code in (("home", HOME), ("away", AWAY)):
            is_team = (nearest == code).astype(float)
            known = (nearest != NOBODY).astype(float)
            numerator = is_team.rolling(window, min_periods=1).sum()
            denominator = known.rolling(window, min_periods=1).sum()
            table["nearest_{}_share_{}f".format(team, window)] = (
                numerator / denominator.replace(0, np.nan)
            )

    # Number of consecutive frames for which the same nearest team was observed
    run_length = np.zeros(len(nearest), dtype=np.int32)
    previous = NOBODY
    current_run = 0
    for i, code in enumerate(nearest.to_numpy()):
        if code == NOBODY:
            current_run = 0
        elif code == previous:
            current_run += 1
        else:
            current_run = 1
        run_length[i] = current_run
        previous = code
    table["nearest_team_run_frames"] = run_length


# Smart tagging data
def normalise_code(value):
    return (value or "").strip().upper()


def tag_team(value):
    """Convert a Smart Tagging team name to HOME/AWAY/NOBODY."""
    name = (value or "").strip().casefold()
    if name in {v.casefold() for v in TAG_NO_TEAM}:
        return NOBODY
    if name not in TAG_TEAMS:
        raise ValueError("unknown team tag: {!r}".format(value))
    return TAG_TEAMS[name]


def tag_features(events, n_frames):
    """Create frame-level dead-ball, goal, possession-team and defending-team tags."""
    dead = np.zeros(n_frames, dtype=bool)
    goal = np.zeros(n_frames, dtype=bool)
    set_piece_type = np.full(n_frames, "", dtype=object)

    active = {
        (family, team): np.zeros(n_frames, dtype=bool)
        for family in ("poss", "def")
        for team in (HOME, AWAY)
    }

    for event in events:
        start = max(0, int(event["start_frame"]))
        end = min(n_frames, int(event["end_frame"]))
        if start >= end:
            continue

        code = normalise_code(event.get("code"))
        tags = event.get("tags") or {}

        if code == SET_PIECE_CODE:
            dead[start:end] = True
            set_piece_type[start:end] = normalise_code(tags.get("Type"))

        if code == GOAL_CODE:
            goal[start:end] = True

        team = tag_team(tags.get("Team"))
        if team != NOBODY:
            family = "def" if code in DEFENDING_CODES else "poss"
            active[(family, team)][start:end] = True

    out = {
        "tag_dead": dead,
        "tag_set_piece_type": set_piece_type,
        "tag_goal": goal,
    }

    for family in ("poss", "def"):
        home_active = active[(family, HOME)]
        away_active = active[(family, AWAY)]

        # Conflicting simultaneous team tags are left unknown, not arbitrarily assigned
        team = np.full(n_frames, NOBODY, dtype=np.int8)
        team[home_active & ~away_active] = HOME
        team[away_active & ~home_active] = AWAY
        out["tag_team_" + family] = team

    return out


# Frame metadata and feature table
def half_mask(game, n_frames):
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half


def build(game=GAME, data_dir=None, validate_baseline=False):
    """Build the per-frame feature table."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]

    home_name, away_name = team_names(meta_path)
    track_teams = team_of_each_track(meta_path)

    print("teams: H = {}, A = {}".format(home_name, away_name))
    print("reading {} ...".format(os.path.basename(raw_path)))
    players, ball = read_atd(raw_path, n_frames, len(track_teams), pitch)

    table = pd.DataFrame({"frame": np.arange(n_frames)})
    table["half"] = half_mask(game, n_frames)
    table["graded"] = table["half"] > 0

    table["ball_x"] = ball[:, 0]
    table["ball_y"] = ball[:, 1]
    table["ball_missing"] = ~np.isfinite(ball[:, 0]) | ~np.isfinite(ball[:, 1])

    length, width = pitch
    table["ball_out"] = (
        (ball[:, 0] < -OUT_MARGIN_M)
        | (ball[:, 0] > length + OUT_MARGIN_M)
        | (ball[:, 1] < -OUT_MARGIN_M)
        | (ball[:, 1] > width + OUT_MARGIN_M)
    )

    (
        table["ball_speed"],
        table["ball_speed_smooth"],
        table["ball_accel"],
    ) = ball_motion(ball, fps)

    spatial_features = nearest_by_team(players, ball, track_teams)
    for name, values in spatial_features.items():
        table[name] = values

    table["nearest_team"] = nearest_team(
        table["d_home"].to_numpy(),
        table["d_away"].to_numpy(),
    )
    table["margin"] = table["d_away"] - table["d_home"]

    (
        table["frames_since_ball_seen"],
        table["last_ball_x"],
        table["last_ball_y"],
    ) = last_seen(ball)

    add_temporal_features(table)

    events = load_smart_tagging.load(game, data_dir)
    for name, values in tag_features(events, n_frames).items():
        table[name] = values

    if validate_baseline:
        # Optional: this rereads the ATD file. Useful once to check that the
        # vectorised nearest-team calculation agrees with the published baseline.
        baseline_labels = nearest_player_labels(
            raw_path, track_teams, n_frames, pitch
        )
        comparable = (baseline_labels != NOBODY) | (
            table["nearest_team"].to_numpy() != NOBODY
        )
        agreement = (
            np.mean(
                baseline_labels[comparable]
                == table["nearest_team"].to_numpy()[comparable]
            )
            if comparable.any()
            else np.nan
        )
        print("nearest-team agreement with baseline: {:.3%}".format(agreement))

    return table


def summarise(table):
    """Print a compact overview of the extracted features."""
    graded = table.loc[table["graded"]]
    print("\n{} frames; {} graded".format(len(table), len(graded)))

    for column in (
        "ball_missing",
        "ball_out",
        "tag_dead",
        "tag_goal",
    ):
        if column in graded:
            print("  {:<22} {:6.1%}".format(column, graded[column].mean()))

    print("\nFeature groups:")
    groups = {
        "Ball position/motion": [
            "ball_x", "ball_y", "ball_speed", "ball_speed_smooth", "ball_accel"
        ],
        "Player proximity": [
            "d_home", "d_away", "margin", "n_home_near", "n_away_near",
            "n_home_very_near", "n_away_very_near",
        ],
        "Tracking availability": ["n_home_tracked", "n_away_tracked"],
        "Temporal context": [
            c for c in table.columns
            if "change_" in c or "share_" in c or c == "nearest_team_run_frames"
        ],
        "Smart Tagging": [
            c for c in table.columns if c.startswith("tag_")
        ],
    }
    for group, columns in groups.items():
        print("  {}: {}".format(group, ", ".join(columns)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--validate-baseline", action="store_true")
    args = parser.parse_args()

    table = build(game=args.game, validate_baseline=args.validate_baseline)

    #parquet_path = OUT_PATH.format(args.game, "parquet")
    csv_path = OUT_PATH.format(args.game, "csv")
    #os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    #table.to_parquet(parquet_path, index=False)
    table.to_csv(csv_path, index=False)

    print("\nWrote:")
    #print("  {}".format(os.path.relpath(parquet_path, ROOT)))
    print("  {}".format(os.path.relpath(csv_path, ROOT)))
    summarise(table)


if __name__ == "__main__":
    main()
