# Preparing the data

NEST ships annotations and metadata only. Source videos must be
downloaded from their original corpora (ChildPlay → YouTube; CHILDES →
TalkBank) and trimmed into per-segment clips. The code lives under
`preprocessing/` and is invoked from the repo root.

All outputs use the **NEST-anonymized identifiers** (`video_id` like
`childplay0023`, `segment_id` like `childplay0023_seg01`).

## Setup

The main `setup_env.sh` already installs everything the preprocessing
scripts need (`yt-dlp`, `ffmpeg`, plus `deno` — yt-dlp's JS-challenge
solver). Activate the env and put `deno` on your PATH:

```bash
conda activate nest
export PATH="$HOME/.deno/bin:$PATH"             # add to ~/.bashrc too
```

## Run (from repo root)

```bash
# 1. Download source videos.
python preprocessing/download/download_videos_youtube.py --cookies cookies.txt
python preprocessing/download/download_videos_childes.py

# 2. Extract per-segment clips (writes both audio and silent versions by default).
python preprocessing/download/extract_segments_from_videos.py
```

Each script skips work already on disk, so re-runs only fill gaps.

## Output layout

```
data/
  videos/
    childplay/<video_id>.mp4           # full source videos
    childes/<video_id>.mp4
  segments/
    childplay/audio/<segment_id>.mp4   # 12s clips, with audio
    childplay/silent/<segment_id>.mp4  # 12s clips, no audio
    childes/audio/<segment_id>.mp4
    childes/silent/<segment_id>.mp4
```

`segments/<dataset>/silent/` is what the evaluation reads in visual mode;
`audio/` is consumed by `nest_eval/transcribe.py` for audio modes.

## ChildPlay (YouTube) authentication

YouTube blocks unauthenticated traffic from datacenter IPs as bot
activity, so on most servers `yt-dlp` needs cookies from a logged-in
YouTube session:

1. In a browser logged into YouTube, install a "cookies.txt" exporter
   extension (e.g. *Get cookies.txt LOCALLY* for Chrome/Firefox).
2. Visit youtube.com, click the extension, save `cookies.txt`.
3. Pass it via `--cookies cookies.txt` (or `export NEST_COOKIES_FILE=...`):

```bash
python preprocessing/download/download_videos_youtube.py --cookies cookies.txt
```

## CHILDES authentication

TalkBank requires registration and acceptance of a data-use agreement
for several CHILDES studies. After registering at
<https://talkbank.org/>:

```bash
export TALKBANK_USER=your-username
export TALKBANK_PASS=your-password
python preprocessing/download/download_videos_childes.py
```

## Input CSVs (shipped under `preprocessing/mappings/`)

| File | Used by | Description |
|---|---|---|
| `video_mapping_index.csv` | `download_videos_youtube.py`, `download_videos_childes.py` | Source-of-truth NEST↔original mapping (`video_id`, `dataset`, `source_video_id`, `source_url`). For ChildPlay rows, `source_video_id` is the YouTube id. |

`extract_segments_from_videos.py` does not read a static CSV — it pulls
segment frame ranges live from `mahuynh/NEST` on HuggingFace, so any
update to NEST is reflected on the next run.

## Known caveats

- **3 ChildPlay segments** (`childplay0067_seg01`, `childplay0077_seg01`,
  `childplay0077_seg02`) reference YouTube videos that have since been
  removed and cannot be downloaded; they are skipped in
  `extract_segments_from_videos.py`.
- **7 ChildPlay segments** carry over a 1-frame extraction lag from the
  original ChildPlay-gaze release; the extractor compensates
  automatically (see `SPECIAL_NEST_SEGMENTS` in
  `extract_segments_from_videos.py`).
