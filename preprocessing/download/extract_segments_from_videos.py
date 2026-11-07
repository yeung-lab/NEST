"""Extract per-segment clips from full source videos.

Pulls segment frame ranges live from the published HuggingFace dataset
(mahuynh/NEST) — no static CSV is required. Handles both ChildPlay and
CHILDES segments uniformly.

Inputs:
  data/videos/<dataset>/<video_id>.mp4   (downloaded by download_videos_youtube.py
                                          or download_videos_childes.py)

Outputs (NEST schema):
  data/segments/<dataset>/audio/<segment_id>.mp4    # with --mode audio  or both
  data/segments/<dataset>/silent/<segment_id>.mp4   # with --mode silent or both

Run from the repo root:
    python preprocessing/download/extract_segments_from_videos.py                        # both audio + silent (default)
    python preprocessing/download/extract_segments_from_videos.py --mode audio           # audio only
    python preprocessing/download/extract_segments_from_videos.py --dataset childplay    # one corpus only

Notes:
- NEST segment_ids in SPECIAL_NEST_SEGMENTS use a 2-frame lag (instead of 1)
  to align with annotations carried over from the original ChildPlay-gaze
  release.
- NEST segment_ids in REMOVED_NEST_SEGMENTS reference YouTube videos that have
  been taken down; these are skipped here. Contact the authors if needed.
"""
import argparse
import json
import os
import shlex
import subprocess as sp

from tqdm import tqdm
from datasets import load_dataset

HF_REPO   = "mahuynh/NEST"
HF_CONFIG = "NEST"
HF_SPLIT  = "test"

# NEST segments that overlap the original ChildPlay-gaze SPECIAL_CLIPS — need 2-frame lag.
SPECIAL_NEST_SEGMENTS = {
    "childplay0004_seg01", "childplay0004_seg02", "childplay0004_seg03",
    "childplay0019_seg01", "childplay0019_seg02",
    "childplay0076_seg01", "childplay0076_seg02",
}
# NEST segments whose source YouTube video has been removed — cannot extract.
REMOVED_NEST_SEGMENTS = {
    "childplay0067_seg01",
    "childplay0077_seg01", "childplay0077_seg02",
}


def get_fps_ffmpeg(video_path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of json "{video_path}"'
    result = sp.run(shlex.split(cmd), capture_output=True, text=True)
    fps_str = json.loads(result.stdout)["streams"][0]["r_frame_rate"]
    num, denom = map(int, fps_str.split("/"))
    return num / denom


def extract_segment(src, dst, time_shift, duration, fps, with_audio):
    """Run ffmpeg to extract one segment. Audio is included iff with_audio."""
    audio_args = "-c:a aac -b:a 128k" if with_audio else "-an"
    cmd = (
        f'ffmpeg -loglevel error -y -i "{src}" -ss {time_shift:.4f} -t {duration:.4f} '
        f'-c:v libx264 -pix_fmt yuv420p -crf 24 {audio_args} -r {fps} '
        f'-movflags +faststart "{dst}"'
    )
    return sp.run(shlex.split(cmd)).returncode == 0


def load_segments(dataset_filter=None, hf_repo=HF_REPO):
    """Return one row per (segment_id) from HF. Filters by dataset if given."""
    print(f"Loading segments from HuggingFace ({hf_repo})...")
    ds = load_dataset(hf_repo, HF_CONFIG, split=HF_SPLIT)
    df = ds.to_pandas()
    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]
    df = df.drop_duplicates(subset="segment_id")[
        ["segment_id", "video_id", "dataset", "start_frame", "end_frame"]
    ].reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_folder", default="data/videos",
                        help="Root folder containing <dataset>/<video_id>.mp4 source videos (default: data/videos)")
    parser.add_argument("--clip_folder",  default="data/segments",
                        help="Output root; <dataset>/(audio|silent) subfolders are created beneath (default: data/segments)")
    parser.add_argument("--dataset",       choices=["childplay", "childes"], default=None,
                        help="Restrict to one source corpus (default: both)")
    parser.add_argument("--mode", choices=["audio", "silent", "both"], default="both",
                        help="Which extraction(s) to produce (default: both)")
    parser.add_argument("--dataset-repo", default=HF_REPO,
                        help=f"HF dataset repo to pull segment metadata from (default: {HF_REPO}).")
    args = parser.parse_args()

    do_audio  = args.mode in ("audio",  "both")
    do_silent = args.mode in ("silent", "both")

    df = load_segments(dataset_filter=args.dataset, hf_repo=args.dataset_repo)
    print(f"Loaded {len(df)} unique segments")

    n_audio = n_silent = n_skip = n_miss = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        seg_id, vid_id, ds_name = row["segment_id"], row["video_id"], row["dataset"]
        if seg_id in REMOVED_NEST_SEGMENTS:
            n_skip += 1
            continue
        src = os.path.join(args.video_folder, ds_name, f"{vid_id}.mp4")
        if not os.path.exists(src):
            n_miss += 1
            continue

        og_fps = round(get_fps_ffmpeg(src))
        frame_shift = 2 if seg_id in SPECIAL_NEST_SEGMENTS else 1
        time_shift = (int(row["start_frame"]) - frame_shift) / og_fps
        duration   = (int(row["end_frame"]) - int(row["start_frame"])) / og_fps

        audio_dir  = os.path.join(args.clip_folder, ds_name, "audio")
        silent_dir = os.path.join(args.clip_folder, ds_name, "silent")
        if do_audio:
            os.makedirs(audio_dir, exist_ok=True)
            dst = os.path.join(audio_dir, f"{seg_id}.mp4")
            if not os.path.exists(dst):
                if extract_segment(src, dst, time_shift, duration, og_fps, with_audio=True):
                    n_audio += 1
        if do_silent:
            os.makedirs(silent_dir, exist_ok=True)
            dst = os.path.join(silent_dir, f"{seg_id}.mp4")
            if not os.path.exists(dst):
                if extract_segment(src, dst, time_shift, duration, og_fps, with_audio=False):
                    n_silent += 1

    print(f"\n=== summary ===")
    if do_audio:  print(f"  audio  written: {n_audio}")
    if do_silent: print(f"  silent written: {n_silent}")
    print(f"  skipped (removed YouTube videos): {n_skip}")
    print(f"  source video missing on disk:     {n_miss}")
    print(f"  output root: {args.clip_folder}")


if __name__ == "__main__":
    main()
