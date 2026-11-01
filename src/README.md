# extract features

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