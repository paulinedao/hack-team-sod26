# extract features from the data atd tracking and the smart tagging

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import AWAY, HOME, NOBODY, team_of_each_track
from starter import load_smart_tagging

NEAR_RADIUS_M = 5.0
OUT_MARGIN_M = 0.5
SPEED_SMOOTH_FRAMES = 5
CONTROL_DISTANCE_M = 1.5

OUT_PATH = os.path.join(DATA_DIR, "derived", "{}_features.parquet")

FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","

TAG_TEAMS = {"barcelona": HOME, "atletic club": AWAY}
TAG_NO_TEAM = "n/a"
SET_PIECE_CODE = "set_pieces"
GOAL_CODE = "goal"
DEFENDING_CODES = ("defending in defensive third", "defending in middle third",
             "defending in attacking third", "defensive transition")

def read_atd(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan)
    
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip().split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1 #frames id counts from 1 and submissions from 0
            
            values = np.array(parts[1].rstrip(GROUP_SEP).replace(GROUP_SEP, VALUE_SEP)).split(VALUE_SEP), dtype=np.float32)
            
            if values.size % n_slots:
                raise ValueError("frame_id {}: {} valuese for {} slots".format(
                    parts[0], values.size, n_slots
                ))
            players[frame] = values.reshape(n_slots, -1)[:, :2]
            
            ball_values = parts[2].rstrip(GROUP_SEP).split(VALUE_SEP)
            ball[frame] = float(ball_values[0]), float(ball_values[1])
            
    scale = np.array([length, width])
    return players * scale.astype(np.float32), ball * scale

def nearest_by_team(players, ball, team_of_track):
    distance = np.hypot(players[...,0] - ball[:, None, 0], players[...,1] - ball[:, None, 1])
    tracked = ~np.isnan(players[...,0])
    out = {}
    for name, team in (("home", HOME), ("away", AWAY)):
        mine = team_of_track == team
        out["d_" + name] = np.fmin.reduce(distance[:, mine], axis=1).astype(np.float32)
        out["n_{}_near".format(name)] = np.sum(distance[:, mine] < NEAR_RADIUS_M, axis=1)
        out["n_{}_tracked".format(name)] = np.sum(tracked[:, mine], axis=1)
    return out
       
def nearest_team(d_home, d_away):
    has_home, has_away = ~np.isnan(d_home), ~np.isnan(d_away)
    out = np.full(len(d_home), NOBODY, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out[has_home & (~has_away | (d_home <= d_away))] = HOME
        out[has_away & (~has_home | (d_away <= d_home))] = AWAY
    return out
    
def ball_motion(ball, fps):
    """ball_speed, ball_speed_smooth, ball_acceel. NaN when either neighbour is missing"""
    step = np.full(len(ball), np.nan)
    step[1:] = np.hypot(*np.diff(ball, axis=0).T)
    speed = step * fps
    smooth = (pd.Series(speed).rolling(SPEED_SMOOTH_FRAMES, center=True, min_periods=1).median().to_numpy(copy=True))
    accel = np.full(len(ball), np.nan)
    accel[1:] = np.diff(smooth) * fps
    return speed, smooth, accel
    
def last_seen(ball):
    """frames_since_ball_seem, last_ball_x, last_ball_y"""
    index = np.arange(len(ball))
    seen = ~np.isnan(ball[:,0])
    last = np.maximum.accumulate(np.where(seen, index, -1))
    known = last - 1
    since = np.where(known, index - last, -1)
    position = np.full_like(ball, np.nan)
    position[known] = ball[last[known]]
    return since, position[:,0], position[:,1]
    
    
def tag_team(value):
    value = (value or TAG_NO_TEAM).strip().lower()
    if value == TAG_NO_TEAM:
        return NOBODY
    if value not in TAG_NO_TEAM:
        raise ValueError("unknown team tag: {}".format(value)) 
    return TAG_TEAMS[value]
    

def tag_features(events, n_frames):
    """tag_dead, tag_set_piece_type, tag_goal, tag_team_posse, tag_team_def"""
    dead = np.zeros(n_frames, dtype=np.bool)
    goal = np.zeros(n_frames, dtype=np.bool)
    set_piece_type = np.full(n_frames, "", dtype=np.object)
    
    active = {(family, team): np.zeros(n_frames, dtype=np.bool) for family in ("poss", "def") for team in (HOME, AWAY)}  
    
    for event in events:
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if start >= end:
            continue
        code = (event["code"] or "").strip().lower()
        if code == SET_PIECE_CODE:
            dead[start:end] = True
            set_piece_type[start:end] = (event["tags"].get("Type") or "").strip().lower()
        if code in GOAL_CODE:
            goal[start:end] = True
        team = tag_team(event["tags"].get("Team"))
        if team != NOBODY:
            family = "def" if code in DEFENDING_CODES else "poss"
            active[(family, team)][start:end] +=1
            
    out = {
        "tag_dead": dead,
        "tag_set_piece_type": set_piece_type,
        "tag_goal": goal}
    
    for family in ("poss", "def"):
        home ,away = active[family, HOME] >0, active[family, AWAY] >0
        team = np.full(n_frames, NOBODY, dtype=np.int8)
        team[home & away] = HOME
        team[away & ~home] = AWAY
        out["tag_team_" + family] = team
    return out

def half_mask(game, n_frames):
    half = np.zeros(n_frames, dtype=np.int8)
    for number,first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half

def build(game=GAME, data_dir=None):
    """The feature table as a df"""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)   
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    length, width = pitch
    
    team_of_track = team_of_each_track(meta_path)
    print("reading {}...".format(os.path.basename(raw_path)))
    players, ball = read_atd(raw_path, n_frames, len(team_of_track), pitch)
    
    table = pd.DataFrame({"frame": np.arange(n_frames)})
    table["half"] = half_mask(game, n_frames)
    table["graded"] = table["half"] > 0
    
    table["ball_x"], table["ball_y"] = ball[:, 0], ball[:, 1]
    table["ball_missing"] = np.isnan(ball[:,0])
    with np.errstate(invalid="ignore"):
        table["ball_out"] = ((ball[:,0] < -OUT_MARGIN_M) | (ball[:,0] > length + OUT_MARGIN_M) | 
                            (ball[:,1] < -OUT_MARGIN_M) | (ball[:,1] > width + OUT_MARGIN_M))
    (table["ball_speed"], table["ball_speed_smooth"], table["ball_accel"]) = ball_motion(ball, GAMES[game]["fps"])
    
    for name, values in nearest_by_team(players, ball, team_of_track).items():
        table[name] = values
    table["nearest_team"] = nearest_team(table["d_home"].to_numpy(), table["d_away"].to_numpy())
    table["margin"] = table["d_away"] - table["d_home"]
    
    (table["frames_since_ball_seen"], table["last_ball_x"], table["last_ball_y"]) = last_seen(ball)
    
    for name, values in tag_features(load_smart_tagging.load(game, data_dir), n_frames).items():
        table[name] = values
    return table

def summarise(table, events):
    graded = table[table["graded"]]
    print("\n{} frames, {} graded".format(len(table), len(graded)))
    print("on graded frames:")
    for column in ("ball_missing", "ball_out", "tag_dead", "tag_goal"):
        print("  {:<22} {:6.1%}".format(column, graded[column].mean()))
    for column in ("d_home", "d_away", "ball_speed_smooth", "n_home_tracked", "n_away_tracked"):
        print("  {:<22} {:6.2f} NaN {:6,1%}".format(column, graded[column].median(), graded[column].isna().mean()))
        
    on_ball = (np.fmin(table["d_home"], table["d_away"]) < CONTROL_DISTANCE_M).to_numpy()
    nearest = table["nearest_team"].to_numpy()
    agree, total = {}, {}
    for event in events:
        team = tag_team(event["tags"].get("Team"))
        if team == NOBODY:
            continue
        frames = np.arange(max(0, event["start_frame"]), min(len(table), event["end_frame"]))
        frames = frames[on_ball[frames]]
        code = event["code"]
        agree["code"] = agree.get(code, 0) + int(np.sum(nearest[frames] == team))
        total["code"] = total.get(code, 0) + len(frames)
    print("\nSmart Tagging Team tag vs nearest team, ball within {} m of a player:".format(CONTROL_DISTANCE_M))
    for code in sorted(total, key=lambda c: -total[c]):
        if total[code]:
            print(" {:<32} agrees {:6.1%} over {:>6} frames".format(code, agree[code] / total[code], total[code]))
            
if __name__ == "__main__":
    table = build()
    parquet_path, csv_path = OUT_PATH.format(GAME, "parquet"), OUT_PATH.format(GAME, "csv")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    table.to_parquet(parquet_path, index=False)
    table.to_csv(csv_path, index=False)
    for path in (parquet_path, csv_path):
        print("wrote {}".format(os.path.relpath(path, ROOT)))
    summarise(table, load_smart_tagging.load())