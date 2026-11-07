"""Download ChildPlay source videos (YouTube) into NEST-schema layout.

Reads mappings/video_mapping_index.csv (rows where dataset == 'childplay') and
saves each video as data/videos/childplay/<video_id>.mp4. <video_id> is the
NEST-anonymized id; the YouTube id is read from the source_video_id column.

Run from the repo root:
    python preprocessing/download/download_videos_youtube.py
    python preprocessing/download/download_videos_youtube.py --cookies cookies.txt    # if YouTube blocks unauthenticated traffic

Cookies (optional): exported from a logged-in YouTube browser session via a
"Get cookies.txt" extension. Required on most servers since YouTube blocks
datacenter IPs as bot traffic. See README for details.
"""
import argparse
import asyncio
import os

import pandas as pd
import yt_dlp


async def download_youtube_video(video_id, youtube_id, output_path=".", cookies=None):
    """Download a single YouTube video, saved as <output_path>/<video_id>.mp4."""
    try:
        os.makedirs(output_path, exist_ok=True)
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        filepath = os.path.join(output_path, f"{video_id}.mp4")
        if os.path.exists(filepath):
            return f"[skip] {video_id} already exists"

        ydl_opts = {
            "outtmpl": f"{output_path}/{video_id}.%(ext)s",
            "format":  "bv*[ext=mp4][vcodec^=avc]+ba[ext=m4a]/b[ext=mp4]",
            "merge_output_format": "mp4",
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }],
        }
        if cookies:
            ydl_opts["cookiefile"] = cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"[ok]   {video_id} ({youtube_id})"
    except Exception as e:
        return f"[fail] {video_id} ({youtube_id}): {e}"


async def download_all_videos(rows, output_path=".", cookies=None):
    return await asyncio.gather(*(
        download_youtube_video(r.video_id, r.youtube_id, output_path, cookies)
        for r in rows.itertuples(index=False)
    ))


def main(args):
    df = pd.read_csv(args.index_csv_path)
    if "video_id" not in df.columns or "source_video_id" not in df.columns:
        raise SystemExit("video_mapping_index.csv must have columns 'video_id' and 'source_video_id'")
    df = df[df["dataset"] == "childplay"].rename(columns={"source_video_id": "youtube_id"})
    print(f"=== {len(df)} ChildPlay videos to fetch ===")
    results = asyncio.run(download_all_videos(df, args.output_folder, args.cookies))

    counts = {"ok": 0, "skip": 0, "fail": 0}
    for r in results:
        tag = r.split("]", 1)[0].lstrip("[")
        counts[tag] = counts.get(tag, 0) + 1
        if tag not in ("ok", "skip"):
            print(r)

    print("\n=== summary ===")
    print(f"  ok         : {counts['ok']}")
    print(f"  skipped    : {counts['skip']}  (already on disk)")
    print(f"  failed     : {counts['fail']}")
    print(f"  output dir : {args.output_folder}/")


_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CSV = os.path.join(_REPO_ROOT, "mappings", "video_mapping_index.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NEST ChildPlay source videos from YouTube")
    parser.add_argument("--index_csv_path", type=str, default=_DEFAULT_CSV,
                        help="Path to video_mapping_index.csv (default: preprocessing/mappings/video_mapping_index.csv)")
    parser.add_argument("--output_folder",  type=str, default="data/videos/childplay",
                        help="Where to save <video_id>.mp4 (default: data/videos/childplay, relative to CWD)")
    parser.add_argument("--cookies",        type=str, default=os.environ.get("NEST_COOKIES_FILE"),
                        help="Netscape cookies.txt for yt-dlp (or NEST_COOKIES_FILE env var)")
    args = parser.parse_args()
    main(args)
