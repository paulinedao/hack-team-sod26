# Feature extraction and possession prediction

To execute the code, please make sure you have the right environment:
```
uv init
uv add numpy pandas
```

The data is located in `/data`, `/data/tactical`and `data/broadcast`.


In the root project, run the code in the terminal using:
```
uv run src/extract_features.py
```

The script is extracting features related to players and the ball: player proximity, temporal features and ball motion. The output csv will appear in `/data/derived`.


### 1. Metadata
- `frame`
- `half`
- `graded`

### 2. Ball position
- `ball_x`
- `ball_y`

### 3. Ball visibility
- `ball_missing`
- `ball_out`

### 4. Ball motion
- `ball_speed`: Ball displacement between consecutive frames multiplied by FPS (in m/s)
- `ball_speed_smooth`:  Rolling median of ball speed over 5 frames in (m/s) 
- `ball_accel`: Change in smoothed ball speed multiplied by FPS (m/s²)

### 5. Player proximity
- `d_home`: Minimum distance between the ball and any tracked home-team player (m) 
- `d_away`: Minimum distance between the ball and any tracked away-team player (m)
- `margin`: `d_away - d_home` (positive means home is closer; negative means away is closer)

### 6. Player counts

Cut off to verify, 5m and 2m

- `n_home_near`: Number of home players within 5 m of the ball 
- `n_away_near`: Number of away players within 5 m of the ball
- `n_home_very_near`: Number of home players within 2 m of the ball
- `n_away_very_near`: Number of away players within 2 m of the ball

### 7. Player tracking availability
- `n_home_tracked`
- `n_away_tracked`

### 8. Nearest team
- `nearest_team`

### 9. Ball history
- `frames_since_ball_seen`
- `last_ball_x`
- `last_ball_y`

### 10. Temporal distance trends
- `d_home_change_10f`: Change in minimum home-player distance over the previous 10 frames
- `d_away_change_10f`: Change in minimum away-player distance over the previous 10 frames

At 25 frames per second (FPS):
- `10f` = 0.4 seconds


### 11. Nearest-team persistence
- `nearest_home_share_25f`: Proportion of the previous 25 frames where the home team was nearest to the ball 
- `nearest_away_share_25f`:Proportion of the previous 25 frames where the away team was nearest to the ball
- `nearest_home_share_50f`: Proportion of the previous 50 frames where the home team was nearest to the ball
- `nearest_away_share_50f`: Proportion of the previous 50 frames where the away team was nearest to the ball
- `nearest_team_run_frames`: Number of consecutive frames for which the current nearest-team label has remained unchanged

At 25 frames per second (FPS):
- `25f` = 1 second
- `50f` = 2 seconds

### 12. Smart Tagging features
- `tag_dead`
- `tag_set_piece_type`
- `tag_goal`
- `tag_team_poss`
- `tag_team_def`


## Function-by-function summary

### `read_atd(raw_path, n_frames, n_slots, pitch)`

- Output: `(players, ball)`
- `players` = `(n_frames, n_slots, 2)` float array of player x/y positions in metres
- `ball` = `(n_frames, 2)` float array of ball x/y positions in metres
- Missing positions are `NaN`
- It reads the raw ATD text file and converts normalized ATD coordinates to real pitch coordinates using the pitch dimensions.

### `nearest_by_team(players, ball, team_of_track)`

- Output: a dictionary of arrays
- Keys like:
  - `d_home`, `d_away`
  - `n_home_near`, `n_away_near`
  - `n_home_very_near`, `n_away_very_near`
  - `n_home_tracked`, `n_away_tracked`
- Meaning:
  - `d_home`: distance from the ball to the nearest home-team player, in metres
  - `d_away`: same for the away team
  - `n_home_near`: number of home players within 5.0 m of the ball
  - `n_home_very_near`: number within 2.0 m
  - `n_home_tracked`: how many home players were actually tracked at that frame
- Units:
  - distances in metres
  - counts as integers

### `nearest_team(d_home, d_away)`

- Output: `np.ndarray` of team labels, shape `(n_frames,)`
- Represents: which team had the closest player to the ball at each frame
- Possible values: `HOME`, `AWAY`, `NOBODY`
- This is a label/code, not a text name.
- Tie-break rule: the baseline’s HOME preference is used when distances are equal.

### `ball_motion(ball, fps)`

- Output: `(speed, smooth_speed, acceleration)`
- `speed`: ball speed in metres per second (`m/s`)
- `smooth_speed`: smoothed ball speed in `m/s`
- `acceleration`: change in ball speed in metres per second squared (`m/s^2`)
- `NaN` is kept when the ball is missing or not enough data exists.
- `fps` is frames-per-second, so the code converts frame-to-frame changes into real-world units.

### `last_seen(ball)`

- Output: `(frames_since, last_ball_x, last_ball_y)`
- `frames_since`: how many frames ago the ball was last seen, as an integer
  - `0` = seen on this frame
  - `1` = last seen one frame ago
- `last_ball_x` and `last_ball_y`: the last known ball position in metres
- This is useful when the ball disappears from the tracking for a few frames.

### `add_temporal_features(table)`

for home change:
A positive value means the ball is farther from the nearest home player now than before.
A negative value means the nearest home player is now closer.

- Output: modifies the DataFrame in place
- Adds features such as:
  - `d_home_change_10f`, `d_away_change_10f`
    - change in nearest-player distance over the last 10 frames
    - unit: metres
  - `nearest_home_share_25f`, `nearest_home_share_50f`
    - proportion of recent frames where the nearest team was home
    - unit: fraction from 0 to 1
  - `nearest_team_run_frames`
    - number of consecutive frames the same nearest team has been observed
    - unit: number of frames
- This captures short-term trends and persistence.

### `normalise_code(value)`

- Output: a cleaned uppercase string
- Example: `" set pieces "` → `"SET PIECES"`
- It strips spaces and uppercase-normalizes tagging codes.

### `tag_team(value)`

- Output: one team code: `HOME`, `AWAY`, or `NOBODY`
- Input is a Smart Tagging team name like `"Barcelona"` or `"Athletic Club"`
- This converts text team names into the project’s internal team IDs.
- It raises an error if the tag is unknown.

### `tag_features(events, n_frames)`

- Output: dictionary of frame-level tag arrays
- Main outputs:
  - `tag_dead`: boolean array, whether a frame is in a dead-ball state
  - `tag_set_piece_type`: string for set-piece type, e.g. `"FREE KICK"`, `"THROW IN"`, etc.
  - `tag_goal`: boolean array, whether it was a goal event
  - `tag_team_poss`: team code for possession-tagged frames
  - `tag_team_def`: team code for defending-tagged frames
- So these are frame-wise labels derived from Smart Tagging events.
- Units:
  - booleans for flags
  - strings for type names
  - team integers for team labels

### `half_mask(game, n_frames)`

- Output: an integer array of shape `(n_frames,)`
- Each element is the half number of the match
- Example:
  - `0` = first half
  - `1` = second half
  - or more detailed block IDs depending on the game
- This is just a frame-level match-half label.

### `build(game=GAME, data_dir=None, validate_baseline=False)`

- Output: a pandas DataFrame with one row per frame
- This is the main “feature table” for the analysis.
- It builds:
  - frame id
  - half info
  - ball position and motion
  - distance-to-nearest-player features
  - nearest-team labels
  - temporal trend features
  - Smart Tagging features
- This is the dataset you use downstream for modeling or possession prediction.
- Units are mixed:
  - positions in metres
  - speeds in m/s
  - distances in metres
  - booleans and counts
  - categorical labels





# Possession prediction

The current possession model lives in `src/predict_possession.py`. It reads a feature CSV produced by `src/extract_features.py` and writes one possession label for each frame.

## Run the prediction

From the project root:

```bash
uv run python src/predict_possession.py --game BAR-ATH
```

Optional debug output includes the per-frame half and score:

```bash
uv run python src/predict_possession.py --game BAR-ATH --debug
```

By default the script writes to `evaluation/my_results/BAR-ATH_predictions_with_literature_wo_hmm.csv` unless you pass `--out`.

## Output format

The prediction file contains:

```csv
frame,label
0,H
1,H
2,D
```

With `--debug`, the script also writes:

```csv
frame,label,half,score
```

Labels are:

- `H`: home team in possession
- `A`: away team in possession
- `D`: dead ball
- `X`: outside the graded match window (used only for non-graded frames)

The script validates that the output is indexed by contiguous frame numbers from `0` to `N-1`.

## Strategy in the current script

The newest implementation is a transparent, rule-based model with an explainable sequence of decisions.

### 1) Dead-ball detection

A frame is marked as `D` when any dead-ball rule fires:

- a Smart Tagging set-piece or goal frame, when `USE_SMART_TAGGING` is enabled
- the ball has been out of play for at least `BALL_OUT_MIN_S`
- the ball is missing and the last known location was near a boundary for at least `BALL_LOST_DEAD_S`
- the ball has been stationary for long enough, when `DEAD_FROM_STATIONARY_BALL` is enabled
- short alive gaps between two dead-ball runs are bridged with `DEAD_BRIDGE_S`
- graded frames before the second-half kickoff are marked dead when `D_BEFORE_KICKOFF` is enabled

This is intentionally conservative: if there is clear evidence the game has stopped, the frame is labeled as dead.

### 2) Possession score

For each alive frame, the script computes a score in the interval `[-1, 1]`:

- `+1`: home is more likely to be in possession
- `-1`: away is more likely to be in possession
- `0`: neutral / weak evidence

The score combines:

- nearest-player proximity gap between home and away
- persistence: how often each team has been nearest in recent frames
- trend: which team is closing in on the ball faster
- crowding: how many players are near the ball
- optional Smart Tagging priors from possession and defending tags

The script also dampens proximity evidence when the ball is far from every player or moving fast enough to be considered in flight.

### 3) State machine

After a dead-ball phase, the model waits for a strong score before assigning a team:

- if the score magnitude is at least `ENTER_THRESHOLD`, a team is assigned
- once a team is established, the model only switches when the opponent reaches `SWITCH_THRESHOLD`
- otherwise it holds the previous state

This hysteresis prevents flicker between `H` and `A` caused by small score fluctuations.

### 4) Spell cleanup

Short home/away runs are merged into their longer neighbor using `MIN_SPELL_S`:

- `H` or `A` spells shorter than the configured minimum are absorbed
- dead-ball segments are left alone

This removes noise from very brief possession bursts.

### 5) Optional HMM mode

The script includes a Viterbi-based Hidden Markov Model behind `USE_HMM`.

- default: `USE_HMM = False`
- if enabled, the script replaces the smoothing, state machine, and short-spell cleanup with an HMM over `H`, `A`, and `D`
- its emissions use the proximity evidence and dead-ball probabilities, while the transition model uses football priors for possession and stoppage durations

## Key parameter defaults

These are the values defined at the top of `src/predict_possession.py` and the ones to review with the football expert:

- `USE_SMART_TAGGING = False`
- `BALL_OUT_MIN_S = 1.0`
- `BALL_LOST_DEAD_S = 4.0`
- `STATIONARY_MIN_S = 2.0`
- `DEAD_BRIDGE_S = 1.0`
- `W_PROXIMITY = 3.0`
- `W_PERSISTENCE = 2.0`
- `W_TREND = 1.0`
- `W_CROWD = 1.0`
- `ENTER_THRESHOLD = 0.25`
- `SWITCH_THRESHOLD = 0.5`
- `MIN_SPELL_S = 2.0`
- `SCORE_SMOOTH_S = 0.6`
- `USE_HMM = False`

## Evaluation workflow

After writing a prediction CSV, evaluate it with:

```bash
uv run python evaluation/run_evaluation.py --results evaluation/my_results --gt public
```

This compares the result against the public ground truth and reports the possession metrics used by the challenge.

## External validation

The project also includes an external validation pipeline using Sportec / IDSSE tracking data. The idea is to test whether the same feature-extraction and possession logic still works on matches outside the BAR-ATH benchmark when more complete tracking data are available.

The validation workflow is:

```bash
uv run src/external_validation.py check-features # list of columns that don't match
uv run src/external_validation.py prepare --reference-gt <5min ground truth> # ground truth summary per match
uv run src/external_validation.py profile --matches J03WMX # comparison table clean vs external degraded
uv run src/external_validation.py run --seeds 0 1 2
uv run src/external_validation.py run --clean # score per match
```

The prepare command will loop over every match in that dictionary and save each one as its own cached .npz file under `data/external/sportec/` 
These commands check that the feature table is consistent with the array-based path, summarize the available external ground truth, compare the clean and degraded tracking quality against BAR-ATH, and finally score the model on external matches with the same possession metrics used in the challenge.



