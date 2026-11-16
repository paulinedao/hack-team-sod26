# How to run our pipeline

## Data location

Place the challenge dataset in the project root under `data/`, or set the `MTAH_DATA_DIR` environment variable to the folder that contains the match assets.

The expected folder structure is:

```text
data/
├── broadcast/
│   ├── BAR-ATH_broadcast_atd.txt
│   ├── BAR-ATH_broadcast_atd.xml
│   ├── BAR-ATH_broadcast_homography.csv
├── tactical/
│   ├── BAR-ATH_tactical_atd.txt
│   ├── BAR-ATH_tactical_atd.xml
│   ├── BAR-ATH_tactical_homography.csv
│   ├── BAR-ATH_tactical_smart_tagging.xml
└── external/
    └── sportec/
```

The main script uses the tactical source by default. It reads the tactical tracking files (`BAR-ATH_tactical_atd.txt` / `BAR-ATH_tactical_atd.xml`) and the tactical smart-tagging file (`BAR-ATH_tactical_smart_tagging.xml`) from the `data/tactical/` folder. The `data/broadcast/` folder is optional and is not used by the challenge pipeline.

## Environment

The core pipeline in `src/my_model_segment_full.py` only needs:

- `numpy`
- `pandas`

Those packages are already listed in `requirements.txt`.

## Commands to run the pipeline

### Recommended: plain pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/my_model_segment_full.py
```

 The script auto-detects the dataset from the repo's `data/` folder or from `MTAH_DATA_DIR` when it is set.

### Alternative: uv

If you prefer `uv`, use:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run python src/my_model_segment_full.py
```

