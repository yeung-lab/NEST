"""Download CHILDES source videos (TalkBank) into NEST-schema layout.

Reads mappings/video_mapping_index.csv (rows where dataset == 'childes') and
saves each video as data/videos/childes/<video_id>.mp4 using its source_url.
<video_id> is the NEST-anonymized id (e.g. 'childes0188').

TalkBank requires registration and a data-use agreement for several CHILDES
studies. Set credentials via env vars before running (from the repo root):

    export TALKBANK_USER=your-username
    export TALKBANK_PASS=your-password
    python preprocessing/download/download_videos_childes.py
"""
import argparse
import os

import pandas as pd
import requests
from tqdm import tqdm


def download_http(video_id, url, out_path, session):
    if os.path.exists(out_path):
        return f"[skip] {video_id} already exists"
    try:
        with session.get(url, stream=True, timeout=60) as r:
            # TalkBank answers unauthenticated requests either with 401/403 or
            # with a 200 HTML login page; never save the latter as an .mp4.
            is_html = r.headers.get("Content-Type", "").startswith("text/html")
            if r.status_code in (401, 403) or (r.ok and is_html):
                return (f"[auth] {video_id}: TalkBank requires authentication "
                        "(set TALKBANK_USER / TALKBANK_PASS)")
            r.raise_for_status()
            tmp = out_path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            os.replace(tmp, out_path)
        return f"[ok]   {video_id}"
    except Exception as e:
        return f"[fail] {video_id}: {e}"


def main(args):
    df = pd.read_csv(args.index_csv_path)
    df = df[df["dataset"] == "childes"]
    if "source_url" not in df.columns or "video_id" not in df.columns:
        raise SystemExit("video_mapping_index.csv must have columns 'video_id' and 'source_url'")
    df = df.dropna(subset=["source_url"])
    print(f"=== {len(df)} CHILDES videos to fetch ===")

    os.makedirs(args.output_folder, exist_ok=True)
    sess = requests.Session()
    user, pwd = os.environ.get("TALKBANK_USER"), os.environ.get("TALKBANK_PASS")
    if user and pwd:
        sess.auth = (user, pwd)
    else:
        print("  (no TALKBANK_USER / TALKBANK_PASS set — public studies only will succeed)")

    counts = {"ok": 0, "skip": 0, "auth": 0, "fail": 0}
    for _, r in tqdm(df.iterrows(), total=len(df)):
        out_path = os.path.join(args.output_folder, f"{r['video_id']}.mp4")
        msg = download_http(r["video_id"], r["source_url"], out_path, sess)
        tag = msg.split("]", 1)[0].lstrip("[")
        counts[tag] = counts.get(tag, 0) + 1
        if tag not in ("ok", "skip"):
            print(msg)

    print("\n=== summary ===")
    print(f"  ok          : {counts['ok']}")
    print(f"  skipped     : {counts['skip']}  (already on disk)")
    print(f"  auth needed : {counts['auth']}  (set TALKBANK_USER / TALKBANK_PASS)")
    print(f"  failed      : {counts['fail']}")
    print(f"  output dir  : {args.output_folder}/")


_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CSV = os.path.join(_REPO_ROOT, "mappings", "video_mapping_index.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NEST CHILDES source videos from TalkBank")
    parser.add_argument("--index_csv_path", type=str, default=_DEFAULT_CSV,
                        help="Path to video_mapping_index.csv (default: preprocessing/mappings/video_mapping_index.csv)")
    parser.add_argument("--output_folder",  type=str, default="data/videos/childes",
                        help="Where to save <video_id>.mp4 (default: data/videos/childes, relative to CWD)")
    args = parser.parse_args()
    main(args)
